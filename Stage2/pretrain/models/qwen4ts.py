
import os
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
    PEFT_OK = True
except Exception:
    PEFT_OK = False


# =========================
# Utilities
# =========================

def _is_rank0_env() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


def _get_bool(cfg, name: str, default: bool) -> bool:
    v = getattr(cfg, name, default)
    return bool(v) if v is not None else bool(default)


def _get_int(cfg, name: str, default: int) -> int:
    v = getattr(cfg, name, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _get_float(cfg, name: str, default: float) -> float:
    v = getattr(cfg, name, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _maybe_get_base_model(m: nn.Module) -> nn.Module:
    """
    PEFT wrapper compatibility: use underlying base model for module discovery.
    """
    gbm = getattr(m, "get_base_model", None)
    if callable(gbm):
        try:
            return gbm()
        except Exception:
            pass

    bm = getattr(m, "base_model", None)
    if bm is not None:
        mm = getattr(bm, "model", None)
        if isinstance(mm, nn.Module):
            return mm

    return m


def _freeze_all_params(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _safe_resize_token_embeddings(model: nn.Module, new_size: int, mean_resizing: bool = True) -> None:
    """
    HF API compatibility: some versions support mean_resizing, some don't.
    """
    try:
        model.resize_token_embeddings(int(new_size), mean_resizing=bool(mean_resizing))
    except TypeError:
        model.resize_token_embeddings(int(new_size))


def _get_input_output_weights(model: nn.Module) -> Tuple[Optional[torch.nn.Parameter], Optional[torch.nn.Parameter]]:
    """
    Return (input_embedding_weight, output_head_weight) if available.
    Works for HF causal LMs and PEFT-wrapped models.
    """
    inp_w = None
    out_w = None

    # input embeddings
    try:
        emb = model.get_input_embeddings()
        if hasattr(emb, "weight") and isinstance(emb.weight, torch.nn.Parameter):
            inp_w = emb.weight
    except Exception:
        inp_w = None

    # output embeddings / lm_head
    out_emb = None
    try:
        if hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
            out_emb = model.get_output_embeddings()
    except Exception:
        out_emb = None

    if isinstance(out_emb, nn.Linear) and hasattr(out_emb, "weight") and isinstance(out_emb.weight, torch.nn.Parameter):
        out_w = out_emb.weight
    else:
        head = getattr(model, "lm_head", None)
        if isinstance(head, nn.Linear) and hasattr(head, "weight") and isinstance(head.weight, torch.nn.Parameter):
            out_w = head.weight

    return inp_w, out_w


def _detect_tied(inp_w: Optional[torch.nn.Parameter], out_w: Optional[torch.nn.Parameter]) -> bool:
    if inp_w is None or out_w is None:
        return False
    if out_w is inp_w:
        return True
    try:
        return (out_w.data_ptr() == inp_w.data_ptr()) and (out_w.numel() == inp_w.numel())
    except Exception:
        return False


def _try_tie_weights_(model: nn.Module) -> None:
    """
    Best-effort: ensure tie_weights is applied if the model supports it.
    This matters because some LMs tie input/output embeddings by default.
    """
    try:
        if hasattr(model, "tie_weights") and callable(model.tie_weights):
            model.tie_weights()
    except Exception:
        pass


# =========================
# Allowlist grad mask
# =========================

def _register_grad_mask_allowlist_once_(param: torch.nn.Parameter, allow_ids: List[int]) -> None:
    """
    Register a hook ONCE per Parameter to zero grads of ALL rows except allowlist rows.
    Robust even if special tokens live inside original/reserved vocab.
    """
    flag = "_grad_mask_allowlist_hook_registered"
    if getattr(param, flag, False):
        return

    allow = sorted({int(i) for i in allow_ids if int(i) >= 0})
    setattr(param, "_grad_mask_allow_ids", allow)

    cache_flag = "_grad_mask_allowlist_cache"
    setattr(param, cache_flag, {})  # device -> bool mask[rows]

    def _hook(grad: torch.Tensor):
        if grad is None:
            return None
        if grad.ndim != 2:
            return grad

        rows = int(grad.size(0))
        device = grad.device

        cache: Dict[torch.device, torch.Tensor] = getattr(param, cache_flag)
        mask = cache.get(device, None)
        if mask is None or int(mask.numel()) != rows:
            m = torch.zeros((rows,), device=device, dtype=torch.bool)
            for idx in allow:
                if 0 <= idx < rows:
                    m[idx] = True
            cache[device] = m
            mask = m

        # Zero out disallowed rows
        # Use masked_fill to avoid in-place indexing issues on some backends
        grad = grad.masked_fill(~mask[:, None], 0)
        return grad

    param.register_hook(_hook)
    setattr(param, flag, True)


def _enable_train_vocab_rows_allowlist_(
    model: nn.Module,
    allow_ids: List[int],
) -> Dict[str, Any]:
    """
    Enable training ONLY for rows in allow_ids on:
      - input embedding weight
      - output head weight (if exists and not tied)
    Handles tied embeddings:
      - if tied, only enable inp_w trainable to avoid double updates
    """
    inp_w, out_w = _get_input_output_weights(model)
    if inp_w is None:
        raise RuntimeError("train_new_vocab_rows=True but cannot find input embedding weight.")

    _try_tie_weights_(model)
    inp_w, out_w = _get_input_output_weights(model)

    tied = _detect_tied(inp_w, out_w) if (out_w is not None) else False

    inp_w.requires_grad_(True)
    if (out_w is not None) and (not tied):
        out_w.requires_grad_(True)

    _register_grad_mask_allowlist_once_(inp_w, allow_ids)
    if (out_w is not None) and (not tied):
        _register_grad_mask_allowlist_once_(out_w, allow_ids)

    return {"tied": tied, "has_out": out_w is not None, "allow_cnt": len(set(int(i) for i in allow_ids))}


# =========================
# LoRA target modules
# =========================

def _infer_lora_target_modules(base_model: nn.Module) -> List[str]:
    """
    Prefer common projection layer suffixes (Qwen/LLaMA style), else fallback to scanning all linear suffixes.
    """
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = set()

    for name, m in base_model.named_modules():
        if isinstance(m, nn.Linear):
            present.add(name.split(".")[-1])

    hits = [n for n in preferred if n in present]
    if hits:
        return hits

    names = sorted(list(present))
    if "lm_head" in names:
        names.remove("lm_head")
    return names


# =========================
# Parameter counting helpers
# =========================

def _count_requires_grad_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_effective_trainable_params(model: nn.Module) -> int:
    """
    Effective trainable:
      - For allowlist-masked embedding/head weights, count only allowed rows * dim.
      - For other requires_grad params (e.g., LoRA A/B), count full.
      - Avoid double-counting shared storage (tied weights).
    """
    effective = 0
    seen = set()

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue

        try:
            key = (p.data_ptr(), p.numel())
        except Exception:
            key = (id(p), p.numel())

        if key in seen:
            continue
        seen.add(key)

        allow = getattr(p, "_grad_mask_allow_ids", None)
        if allow is not None and p.ndim == 2:
            rows = int(p.size(0))
            dim = int(p.size(1))
            cnt = 0
            for idx in allow:
                if 0 <= int(idx) < rows:
                    cnt += 1
            effective += cnt * dim
        else:
            effective += p.numel()

    return int(effective)


# =========================
# Wrapper Model
# =========================

class Model(nn.Module):
    """
    Wrapper used by Exp_Main.

    Exposed attributes:
      - tokenizer
      - model                 (HF CausalLM or PEFT-wrapped)
      - base_vocab_size       (AFTER adding <TS_START>/<TS_END>, BEFORE codebook expansion)
      - ts_start_id / ts_end_id
      - vocab_size            (= base_vocab_size + codebook_size)
      - codebook_size
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        trust_remote_code = _get_bool(configs, "trust_remote_code", False)
        mean_resizing = _get_bool(configs, "mean_resizing", True)
        want_bf16 = _get_bool(configs, "bf16", True)

        # optional: allow user set attn impl (default sdpa is usually more stable)
        attn_impl = getattr(configs, "attn_implementation", None) or "sdpa"

        # -----------------
        # Tokenizer
        # -----------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            configs.LLM_model_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # Add TS tokens (NOTE: may already exist in reserved vocab; allowlist handles this)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<TS_START>", "<TS_END>"]})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -----------------
        # HF model loading
        # -----------------
        load_kwargs = dict(
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
            device_map=None,
        )
        if want_bf16:
            try:
                hf = AutoModelForCausalLM.from_pretrained(
                    configs.LLM_model_path,
                    torch_dtype=torch.bfloat16,
                    **load_kwargs,
                )
            except TypeError:
                hf = AutoModelForCausalLM.from_pretrained(
                    configs.LLM_model_path,
                    dtype=torch.bfloat16,
                    **load_kwargs,
                )
        else:
            hf = AutoModelForCausalLM.from_pretrained(configs.LLM_model_path, **load_kwargs)

        self.model = hf

        if _is_rank0_env():
            impl = getattr(self.model.config, "_attn_implementation", None) or getattr(self.model.config, "attn_implementation", None)
            print(f"[attn] using impl = {impl}")

        # training default: disable use_cache
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # -----------------
        # Step 1: resize for TS special tokens
        # -----------------
        old_vocab_before_ts = int(self.model.get_input_embeddings().num_embeddings)
        if old_vocab_before_ts < len(self.tokenizer):
            _safe_resize_token_embeddings(self.model, len(self.tokenizer), mean_resizing=mean_resizing)

        # base vocab size AFTER TS tokens (boundary for time tokens)
        self.base_vocab_size = int(self.model.get_input_embeddings().num_embeddings)

        # Resolve TS ids (must be inside base vocab)
        self.ts_start_id = int(self.tokenizer.convert_tokens_to_ids("<TS_START>"))
        self.ts_end_id = int(self.tokenizer.convert_tokens_to_ids("<TS_END>"))

        if self.ts_start_id < 0 or self.ts_end_id < 0:
            raise ValueError("Failed to resolve <TS_START>/<TS_END> token ids.")
        if not (0 <= self.ts_start_id < self.base_vocab_size) or not (0 <= self.ts_end_id < self.base_vocab_size):
            raise ValueError(
                f"Special token ids must be inside base vocab. "
                f"ts_start_id={self.ts_start_id}, ts_end_id={self.ts_end_id}, base_vocab_size={self.base_vocab_size}"
            )

        # -----------------
        # Step 2: expand vocab for codebook tokens
        # -----------------
        self.codebook_size = _get_int(configs, "codebook_size", 0)
        if self.codebook_size <= 0:
            raise ValueError(f"configs.codebook_size must be > 0, got {self.codebook_size}")

        self.vocab_size = int(self.base_vocab_size + self.codebook_size)

        _safe_resize_token_embeddings(self.model, self.vocab_size, mean_resizing=mean_resizing)

        # Ensure model configs see the logical vocab size
        if hasattr(self.model, "config"):
            self.model.config.vocab_size = int(self.vocab_size)
        if hasattr(self.model, "generation_config"):
            try:
                self.model.generation_config.vocab_size = int(self.vocab_size)
            except Exception:
                pass

        # ensure tie_weights after resizing
        _try_tie_weights_(self.model)
        # self.model.init_weights()
        # _try_tie_weights_(self.model)

        # -----------------
        # Freeze all params first
        # -----------------
        _freeze_all_params(self.model)

        # -----------------
        # Optional: LoRA
        # -----------------
        if _get_bool(configs, "lora_enable", False):
            if not PEFT_OK:
                raise RuntimeError("peft not installed but lora_enable=True")

            base_for_search = _maybe_get_base_model(self.model)
            target_modules = _infer_lora_target_modules(base_for_search)
            if len(target_modules) == 0:
                raise RuntimeError("No nn.Linear modules found for LoRA injection.")

            if _is_rank0_env():
                print(f"[lora] target_modules={target_modules}")

            lora_config = LoraConfig(
                r=_get_int(configs, "lora_r", 8),
                lora_alpha=_get_int(configs, "lora_alpha", 16),
                target_modules=target_modules,
                lora_dropout=_get_float(configs, "lora_dropout", 0.0),
                bias=getattr(configs, "lora_bias", "none"),
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)

            if hasattr(self.model, "print_trainable_parameters") and _is_rank0_env():
                self.model.print_trainable_parameters()

        # -----------------
        # Optional: train ONLY allowlisted vocab rows
        # -----------------
        mask_report = None
        if _get_bool(configs, "train_new_vocab_rows", False):
            # allowlist:
            # 1) TS special tokens rows (may be inside original/reserved vocab)
            # 2) codebook rows [base_vocab_size, vocab_size)
            allow_ids = [self.ts_start_id, self.ts_end_id] + list(range(self.base_vocab_size, self.vocab_size))
            mask_report = _enable_train_vocab_rows_allowlist_(self.model, allow_ids=allow_ids)

        # -----------------
        # bf16 cast safety
        # -----------------
        if want_bf16:
            try:
                self.model.to(dtype=torch.bfloat16)
            except Exception:
                pass

        # -----------------
        # rank0: print per-parameter requires_grad
        # -----------------
        # if _is_rank0_env():
        #     for name, param in self.model.named_parameters():
        #         status = "✅ YES" if param.requires_grad else "❄️ NO"
        #         print(f"{name:<100} | {status}")

        # -----------------
        # rank0: print summary
        # -----------------
        if _get_bool(configs, "print_trainable", False) and _is_rank0_env():
            total = _count_total_params(self.model)
            trainable = _count_requires_grad_params(self.model)
            pct = 100.0 * float(trainable) / max(float(total), 1.0)

            inp_w, out_w = _get_input_output_weights(self.model)
            tied = _detect_tied(inp_w, out_w) if (inp_w is not None and out_w is not None) else False

            effective = _count_effective_trainable_params(self.model)
            eff_pct = 100.0 * float(effective) / max(float(total), 1.0)

            extra = ""
            if mask_report is not None:
                extra = f" | mask_applied=True(tied={mask_report['tied']}, has_out={mask_report['has_out']}, allow_cnt={mask_report['allow_cnt']})"

            print(
                f"[qwen4ts-resize] base_vocab={self.base_vocab_size}, codebook={self.codebook_size}, total_vocab={self.vocab_size} | "
                f"tied_inp_out={tied} | "
                f"train_new_vocab_rows={_get_bool(configs,'train_new_vocab_rows',False)} | "
                f"trainable={trainable:,}/{total:,} ({pct:.4f}%) | "
                f"effective_trainable={effective:,}/{total:,} ({eff_pct:.4f}%) | "
                f"bf16={want_bf16}"
                f"{extra}"
            )

    def forward(self, **kwargs):
        return self.model(**kwargs)
