import os
import json
import random
import contextlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from transformers import get_cosine_schedule_with_warmup
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from utils.tools import EarlyStopping
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import qwen4ts
from TStokenizer.models.SVQ import Model as SVQModel


# ----------------- distributed helpers -----------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _sync_dist_info(args) -> None:
    if dist.is_available() and dist.is_initialized():
        args.distributed = True
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)


def _is_distributed(args) -> bool:
    return bool(getattr(args, "distributed", False)) and dist.is_available() and dist.is_initialized()


def _rank(args) -> int:
    return int(dist.get_rank()) if _is_distributed(args) else int(getattr(args, "rank", 0))


def _is_rank0(args) -> bool:
    return (not _is_distributed(args)) or _rank(args) == 0


def _world_size(args) -> int:
    return int(dist.get_world_size()) if _is_distributed(args) else int(os.environ.get("WORLD_SIZE", "1") or 1)

def _barrier(args):
    if _is_distributed(args):
        dist.barrier()


def _all_reduce_sum(t: torch.Tensor, args) -> torch.Tensor:
    if _is_distributed(args):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _unwrap_fsdp(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, FSDP) else m


def _has_fsdp_child(module: nn.Module) -> bool:
    for _, mm in module.named_modules():
        if isinstance(mm, FSDP):
            return True
    return False


# ----------------- small helpers -----------------

def _get_int(args, name: str, default: int) -> int:
    v = getattr(args, name, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _get_float(args, name: str, default: float) -> float:
    v = getattr(args, name, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_bool(args, name: str, default: bool) -> bool:
    v = getattr(args, name, default)
    return bool(v) if v is not None else bool(default)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_bf16() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe is bf16-only and requires CUDA. No GPU detected.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU/driver does not support bf16, but bf16-only was requested.")


def _cast_module_to_bf16_(module: nn.Module) -> None:
    for p in module.parameters(recurse=True):
        if p is not None and p.data is not None and p.data.dtype != torch.bfloat16:
            p.data = p.data.to(dtype=torch.bfloat16)
    for b in module.buffers(recurse=True):
        if b is not None and b.data is not None and b.data.dtype.is_floating_point and b.data.dtype != torch.bfloat16:
            b.data = b.data.to(dtype=torch.bfloat16)


# ----------------- Loss/Acc -----------------

def _ce_loss_mean_and_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=int(ignore_index),
        reduction="mean",
        label_smoothing=float(label_smoothing),
    )

    with torch.no_grad():
        pred = shift_logits.argmax(dim=-1)
        mask = shift_labels.ne(int(ignore_index))
        correct = ((pred == shift_labels) & mask).sum().to(torch.int64)
        total = mask.sum().to(torch.int64)

    return loss, correct, total


# ----------------- Collator + prompt cache -----------------

class PromptLRUCache:
    def __init__(self, max_items: int = 2048):
        self.max_items = int(max_items)
        self._d: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        v = self._d.get(key, None)
        if v is None:
            return None
        self._d.move_to_end(key)
        return v

    def put(self, key: str, value: torch.Tensor) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


class CollatorTS:
    """
    seq = [prompt] <TS_START> x <TS_END> <TS_START> y <TS_END>
    labels supervise only y and the final <TS_END>
    """
    def __init__(
        self,
        tokenizer,
        ts_start_id: int,
        ts_end_id: int,
        base_vocab_size: int,
        codebook_size: int,
        pad_to_multiple_of: int = 8,
        default_prompt_text: Optional[str] = None,
        enable_prompt_cache: bool = True,
        prompt_cache_max_items: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.ts_start_id = int(ts_start_id)
        self.ts_end_id = int(ts_end_id)
        self.offset = int(base_vocab_size)  # time token id = base_vocab + raw_id
        self.codebook_size = int(codebook_size)

        self.pad_id = int(tokenizer.pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else 0

        self.default_prompt_text = default_prompt_text
        self._cache = PromptLRUCache(prompt_cache_max_items) if bool(enable_prompt_cache) else None

        self._ts_start = torch.tensor([self.ts_start_id], dtype=torch.long)
        self._ts_end = torch.tensor([self.ts_end_id], dtype=torch.long)

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        if self._cache is not None:
            hit = self._cache.get(prompt_text)
            if hit is not None:
                return hit
        ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].long().cpu()
        if self._cache is not None:
            self._cache.put(prompt_text, ids)
        return ids

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs: List[torch.Tensor] = []
        labs: List[torch.Tensor] = []

        for it in batch:
            prompt_text = it.get("prompt_text", None) or self.default_prompt_text
            if prompt_text is None:
                raise ValueError("No prompt_text found in sample and default_prompt_text is None.")
            prompt_ids = self._encode_prompt(prompt_text)

            x_raw = it["x_ids_raw"].long().cpu()
            y_raw = it["y_ids_raw"].long().cpu()

            x = x_raw + self.offset
            y = y_raw + self.offset

            seq = torch.cat([prompt_ids, self._ts_start, x, self._ts_end, self._ts_start, y, self._ts_end], dim=0).long()

            lab = seq.clone()
            prompt_len = int(prompt_ids.numel())
            out_start = prompt_len + 1 + int(x.numel()) + 1 + 1
            lab[:out_start] = -100

            seqs.append(seq)
            labs.append(lab)

        B = len(seqs)
        max_len = max(int(s.numel()) for s in seqs)
        if self.pad_to_multiple_of and (max_len % self.pad_to_multiple_of != 0):
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        label_ids = torch.full((B, max_len), -100, dtype=torch.long)

        for i, (s, l) in enumerate(zip(seqs, labs)):
            L = int(s.numel())
            input_ids[i, :L] = s
            label_ids[i, :L] = l
            attn_mask[i, :L] = 1

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": label_ids}


class TimeThenEndProcessor(LogitsProcessor):
    """
    Generate Tout time-tokens, then generate exactly one <TS_END>.

    generated_len = input_ids.shape[1] - prefix_len
    - generated_len <  Tout: only time tokens in [offset, offset+codebook_size)
    - generated_len == Tout: only ts_end_id
    """
    def __init__(self, prefix_len: int, Tout: int, offset: int, codebook_size: int, ts_end_id: int):
        self.prefix_len = int(prefix_len)
        self.Tout = int(Tout)
        self.offset = int(offset)
        self.codebook_size = int(codebook_size)
        self.ts_end_id = int(ts_end_id)

        self._time_mask_cache: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
        self._end_mask_cache: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        if self.Tout <= 0:
            raise ValueError(f"Tout must be > 0, got {self.Tout}")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be > 0, got {self.codebook_size}")

    def _build_time_mask(self, scores: torch.Tensor) -> torch.Tensor:
        V = int(scores.size(-1))
        mask = torch.full((V,), float("-inf"), device=scores.device, dtype=scores.dtype)
        lo = self.offset
        hi = min(self.offset + self.codebook_size, V)
        if lo < 0 or lo >= V:
            raise RuntimeError(f"time offset out of vocab: offset={lo}, vocab={V}")
        mask[lo:hi] = 0.0

        # 防止 <TS_END> 在 time 阶段被允许
        if lo <= self.ts_end_id < hi:
            mask[self.ts_end_id] = float("-inf")

        return mask

    def _build_end_mask(self, scores: torch.Tensor) -> torch.Tensor:
        V = int(scores.size(-1))
        mask = torch.full((V,), float("-inf"), device=scores.device, dtype=scores.dtype)
        if self.ts_end_id < 0 or self.ts_end_id >= V:
            raise RuntimeError(f"ts_end_id out of vocab: ts_end_id={self.ts_end_id}, vocab={V}")
        mask[self.ts_end_id] = 0.0
        return mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        V = int(scores.size(-1))
        generated_len = int(input_ids.size(1)) - self.prefix_len
        if generated_len < 0:
            raise RuntimeError(
                f"generated_len < 0 (prefix_len wrong?): generated_len={generated_len}, prefix_len={self.prefix_len}"
            )

        key = (scores.device, scores.dtype, V)

        if generated_len < self.Tout:
            mask = self._time_mask_cache.get(key)
            if mask is None:
                mask = self._build_time_mask(scores)
                self._time_mask_cache[key] = mask
            return scores + mask

        if generated_len == self.Tout:
            mask = self._end_mask_cache.get(key)
            if mask is None:
                mask = self._build_end_mask(scores)
                self._end_mask_cache[key] = mask
            return scores + mask

        # generated_len > Tout：理论上不会发生（因为 steps=Tout+1），放行
        return scores


# ----------------- Exp_Main -----------------

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        _sync_dist_info(args)

        # seed (rank-aware)
        if getattr(args, "random_seed", None) is not None:
            base = int(args.random_seed)
            _seed_everything(base + _rank(args))

        super().__init__(args)

        self._init_bf16()

    def _init_bf16(self) -> None:
        _require_bf16()
        self.use_gpu = True
        self.use_amp = True
        self.amp_dtype = torch.bfloat16
        self.scaler = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")

    def _autocast_ctx(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def _build_model(self):
        # ---- 多卡 FULL_SHARD 下，默认禁用 train_new_vocab_rows（稳定性/语义风险）----
        ws = _world_size(self.args)
        # if ws > 1 and _get_bool(self.args, "train_new_vocab_rows", False):
        #     if _is_rank0(self.args):
        #         print("[warn] FULL_SHARD + train_new_vocab_rows is unsafe; disabling for stability.")
        #     setattr(self.args, "train_new_vocab_rows", False)

        model_dict = {"qwen4ts": qwen4ts}
        base_model = model_dict[self.args.model].Model(self.args)

        # gradient checkpointing (if HF supports)
        if _get_bool(self.args, "gradient_checkpointing", False):
            inner = getattr(base_model, "model", None)
            if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()
            if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "use_cache"):
                inner.config.use_cache = False

        # bf16 cast
        try:
            base_model.to(dtype=torch.bfloat16)
        except Exception:
            pass
        _cast_module_to_bf16_(base_model)

        enable_auto_wrap = _get_bool(self.args, "fsdp_auto_wrap", False)
        min_params = _get_int(self.args, "fsdp_min_num_params", 1e6)

        def _auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
            return size_based_auto_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                min_num_params=int(min_params),
            )

        auto_wrap = None
        if enable_auto_wrap and (ws > 1) and (not _has_fsdp_child(base_model)):
            auto_wrap = _auto_wrap_policy

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        sharding = ShardingStrategy.NO_SHARD if ws == 1 else ShardingStrategy.FULL_SHARD
        sync_states = _get_bool(self.args, "sync_module_states", False)

        fsdp_model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=sharding,
            mixed_precision=mp,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            sync_module_states=bool(sync_states),
            use_orig_params=True,
            forward_prefetch=False,
        )

        # SVQ decode model (frozen)
        with open(self.args.tokens_config, "r") as f:
            svq_cfg = json.load(f)
        vq = SVQModel(svq_cfg)
        vq.load_state_dict(torch.load(self.args.tokens_checkpoints, map_location="cpu"))
        vq.eval()
        for p in vq.parameters():
            p.requires_grad = False
        self.vq_model = vq.to(self.device)

        tok = _unwrap_fsdp(fsdp_model).tokenizer
        return fsdp_model, tok

    def _select_optimizer(self) -> AdamW:
        lr = _get_float(self.args, "learning_rate", 1e-4)
        wd = _get_float(self.args, "weight_decay", 0.0)
        return AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(wd), foreach=False)

    def _build_scheduler(self, optimizer: AdamW, train_loader_len: int):
        epochs = _get_int(self.args, "train_epochs", 1)
        warmup_ratio = _get_float(self.args, "warmup_ratio", 0.0)
        gas = max(1, _get_int(self.args, "gradient_accumulation_steps", 1))

        steps_per_epoch = (int(train_loader_len) + gas - 1) // gas
        total_steps = max(1, steps_per_epoch * int(epochs))

        warmup_steps = int(total_steps * float(warmup_ratio)) if warmup_ratio > 0 else 0
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    @torch.inference_mode()
    def eval_teacher_forcing(self, loader):
        self.model.eval()

        loss_sum = torch.tensor(0.0, device=self.device)
        tok_total = torch.tensor(0, device=self.device, dtype=torch.int64)
        correct = torch.tensor(0, device=self.device, dtype=torch.int64)

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with self._autocast_ctx():
                out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                loss_mean, c, t = _ce_loss_mean_and_counts(
                    out.logits,
                    labels,
                    ignore_index=-100,
                    label_smoothing=_get_float(self.args, "label_smoothing", 0.0),
                )

            loss_sum += loss_mean.detach() * t.to(loss_sum.dtype)
            tok_total += t
            correct += c

        loss_sum = _all_reduce_sum(loss_sum, self.args)
        tok_total = _all_reduce_sum(tok_total, self.args)
        correct = _all_reduce_sum(correct, self.args)

        mean_loss = float((loss_sum / tok_total.clamp_min(1)).item())
        mean_acc = float((correct.to(torch.float32) / tok_total.clamp_min(1).to(torch.float32)).item())

        self.model.train()
        return mean_loss, mean_acc

    def train_one_epoch(self, loader, optimizer: AdamW, scheduler):
        self.model.train()

        grad_clip = _get_float(self.args, "grad_clip", 0.0)
        gas = max(1, _get_int(self.args, "gradient_accumulation_steps", 1))

        loss_sum = torch.tensor(0.0, device=self.device)
        tok_total = torch.tensor(0, device=self.device, dtype=torch.int64)
        correct = torch.tensor(0, device=self.device, dtype=torch.int64)

        # 关键：知道总 batch 数，保证最后一个 batch 一定同步
        num_batches = len(loader)

        pbar = tqdm(loader, desc="Train", leave=False, disable=(not _is_rank0(self.args)))
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(pbar, start=1):
            is_last = (micro_step == num_batches)

            # 关键：最后一个 batch 强制 sync（即使 micro_step % gas != 0）
            sync_this_step = ((micro_step % gas) == 0) or is_last

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            sync_ctx = contextlib.nullcontext() if sync_this_step else self.model.no_sync()
            with sync_ctx:
                with self._autocast_ctx():
                    out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                    loss_mean, c, t = _ce_loss_mean_and_counts(
                        out.logits,
                        labels,
                        ignore_index=-100,
                        label_smoothing=_get_float(self.args, "label_smoothing", 0.0),
                    )
                    (loss_mean / float(gas)).backward()

            with torch.no_grad():
                loss_sum += loss_mean.detach() * t.to(loss_sum.dtype)
                tok_total += t
                correct += c

            if sync_this_step:
                if grad_clip > 0:
                    FSDP.clip_grad_norm_(self.model, float(grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if _is_rank0(self.args):
                cur_loss = float((loss_sum / tok_total.clamp_min(1)).item())
                cur_acc = float((correct.to(torch.float32) / tok_total.clamp_min(1).to(torch.float32)).item())
                pbar.set_postfix(loss=cur_loss, acc=cur_acc * 100.0)

        # 注意：这里不要再做 for-loop 外的“尾巴 step”
        loss_sum = _all_reduce_sum(loss_sum, self.args)
        tok_total = _all_reduce_sum(tok_total, self.args)
        correct = _all_reduce_sum(correct, self.args)

        mean_loss = float((loss_sum / tok_total.clamp_min(1)).item())
        mean_acc = float((correct.to(torch.float32) / tok_total.clamp_min(1).to(torch.float32)).item())
        return mean_loss, mean_acc

    @torch.inference_mode()
    def _encode_prompts_batch_leftpad(self, prompt_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        old_side = getattr(self.tokenizer, "padding_side", "right")
        try:
            self.tokenizer.padding_side = "left"
            enc = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_side

        prompt_ids = enc["input_ids"].to(self.device, non_blocking=True)
        prompt_mask = enc["attention_mask"].to(self.device, non_blocking=True)
        return prompt_ids, prompt_mask



    @torch.inference_mode()
    def eval_generate_all_distributed(self, raw_test_loader, Tout: int, pred_len: int):
        """
        Generate-eval aligned with training target:
          generate Tout time tokens + 1 final <TS_END> token.

        Metrics:
          Use sklearn-based `metric(pred, true)` exactly as you want:
            mae, mse, rmse, mape, r2 = metric(np.squeeze(preds, -1), np.squeeze(trues, -1))

        Distributed (2 GPUs / multi-rank):
          - token accuracies are reduced by all-reduce sum
          - preds/trues are gathered to rank0 via all_gather_object
          - rank0 computes sklearn metrics
          - metrics are broadcast back to all ranks
        """
        import numpy as np
        import torch
        import torch.distributed as dist
        from tqdm import tqdm
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers.generation.logits_process import LogitsProcessorList


        def _metric_distributed_sklearn(preds_t: torch.Tensor, trues_t: torch.Tensor):
 
            preds_np = preds_t.detach().float().cpu().numpy()
            trues_np = trues_t.detach().float().cpu().numpy()
            preds_np = np.squeeze(preds_np, -1)
            trues_np = np.squeeze(trues_np, -1)
            mae, mse, rmse, mape, r2 = metric(preds_np, trues_np)
            mae_16, mse_16, rmse_16, mape_16, r2_16 = metric(preds_np[:,:16], trues_np[:,:16])

            return float(mae), float(mse), float(rmse), float(mape), float(r2), float(mae_16), float(mse_16), float(rmse_16), float(mape_16), float(r2_16)


        # -------------------------
        # eval mode
        # -------------------------
        self.model.eval()
        self.vq_model.eval()

        um = _unwrap_fsdp(self.model)
        base_vocab = int(um.base_vocab_size)
        ts_start_id = int(um.ts_start_id)
        ts_end_id = int(um.ts_end_id)

        gen_use_cache = _get_bool(self.args, "gen_use_cache", True)
        codebook = int(self.args.codebook_size)

        # token-level accuracies (distributed-friendly)
        correct_time = torch.zeros((), device=self.device, dtype=torch.int64)
        total_time = torch.zeros((), device=self.device, dtype=torch.int64)
        correct_end = torch.zeros((), device=self.device, dtype=torch.int64)
        total_end = torch.zeros((), device=self.device, dtype=torch.int64)
        bad_time_tokens = torch.zeros((), device=self.device, dtype=torch.int64)

        # store preds/trues for sklearn metrics (gather to rank0 at the end)
        pred_buf = []
        true_buf = []

        Tout_i = int(Tout)
        steps = Tout_i + 1  # Tout time tokens + final <TS_END>

        pbar = tqdm(
            raw_test_loader,
            desc=f"Gen-eval rank{_rank(self.args)}",
            disable=(not _is_rank0(self.args))
        )

        for batch in pbar:
            x_all = batch["x_ids_raw"]             # [B, Tin] raw ids in [0, K)
            y_all = batch["y_ids_raw"]             # [B, Tout] raw ids in [0, K)
            y_true_all = batch["batch_y"]          # [B, pred_len] or compatible
            y_stats_all = batch["y_stats"]         # dict
            prompt_all = batch.get("prompt_text", None)
            if prompt_all is None:
                raise ValueError("prompt_text missing in dataset; cannot do variable-prompt generate eval.")

            B = int(x_all.size(0))

            # prompts (left-pad)
            prompt_ids, prompt_mask = self._encode_prompts_batch_leftpad(list(prompt_all))
            # IMPORTANT: ensure on device
            prompt_ids = prompt_ids.to(self.device, non_blocking=True).long()
            prompt_mask = prompt_mask.to(self.device, non_blocking=True).long()

            # x token offset into extended vocab
            x = x_all.to(self.device, non_blocking=True).long() + base_vocab
            Tin = int(x.size(1))

            ones1 = torch.ones((B, 1), device=self.device, dtype=prompt_mask.dtype)
            ts_start = torch.full((B, 1), ts_start_id, device=self.device, dtype=torch.long)
            ts_end = torch.full((B, 1), ts_end_id, device=self.device, dtype=torch.long)

            # prefix ends with <TS_START> right before y generation
            prefix = torch.cat([prompt_ids, ts_start, x, ts_end, ts_start], dim=1).long()
            prefix_attn = torch.cat(
                [prompt_mask,
                 ones1,
                 torch.ones((B, Tin), device=self.device, dtype=prompt_mask.dtype),
                 ones1,
                 ones1],
                dim=1,
            ).long()

            prefix_len = int(prefix.size(1))

            # enforce: first Tout tokens are time tokens, last is <TS_END>
            logits_proc = TimeThenEndProcessor(
                prefix_len=prefix_len,
                Tout=Tout_i,
                offset=base_vocab,
                codebook_size=codebook,
                ts_end_id=ts_end_id,
            )

            inner = _unwrap_fsdp(self.model)
            hf = getattr(inner, "model", None)
            if hf is None or (not hasattr(hf, "generate")):
                raise RuntimeError("Expected a HF CausalLM at inner.model with a .generate() method.")

            # HF generate may need full params under FSDP (your original approach)
            with FSDP.summon_full_params(self.model, recurse=True, writeback=False):
                with self._autocast_ctx():
                    gen = hf.generate(
                        input_ids=prefix,
                        attention_mask=prefix_attn,
                        max_new_tokens=steps,
                        min_new_tokens=steps,
                        do_sample=False,
                        num_beams=1,
                        logits_processor=LogitsProcessorList([logits_proc]),
                        pad_token_id=int(self.tokenizer.pad_token_id),
                        use_cache=bool(gen_use_cache),
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

            # new tokens
            new = gen[:, -steps:]          # [B, Tout+1]
            new_time = new[:, :Tout_i]     # [B, Tout]
            new_end = new[:, Tout_i]       # [B]

            # sanity: all time tokens within [base_vocab, base_vocab+K)
            bad = ((new_time < base_vocab) | (new_time >= base_vocab + codebook)).sum().to(torch.int64)
            bad_time_tokens += bad

            pred_ids_raw = (new_time - base_vocab).clamp(0, codebook - 1).long()

            # token acc (time)
            y_tok = y_all[:, :Tout_i].to(self.device, non_blocking=True).long()
            correct_time += (pred_ids_raw == y_tok).sum().to(torch.int64)
            total_time += B * Tout_i

            # token acc (end)
            correct_end += (new_end == ts_end_id).sum().to(torch.int64)
            total_end += B

            # recon to series
            token_ids = pred_ids_raw.view(B, 1, Tout_i)  # match your vq interface
            token_ids = F.pad(token_ids, (0, 24 - Tout_i), value=0)
            stats_dev = {k: v.to(self.device, non_blocking=True) for k, v in y_stats_all.items()}
            recon = self.vq_model.ids_to_series(token_ids, stats_dev)  # expected numel == B*pred_len

            # shape to [N, pred_len]
            pred_series = recon.reshape(-1, int(pred_len)).to(torch.float32)
            true_series = y_true_all.to(self.device, non_blocking=True).reshape(-1, int(pred_len)).to(torch.float32)

            # (optional) strict shape check
            if pred_series.shape != true_series.shape:
                raise RuntimeError(f"pred/true shape mismatch: pred={tuple(pred_series.shape)} true={tuple(true_series.shape)}")

            # store for sklearn metric; add trailing dim=1 so np.squeeze(...,-1) works
            pred_buf.append(pred_series.unsqueeze(-1))   # [n_i, pred_len, 1]
            true_buf.append(true_series.unsqueeze(-1))   # [n_i, pred_len, 1]

        # -------------------------
        # reduce accuracies across ranks
        # -------------------------
        correct_time = _all_reduce_sum(correct_time, self.args)
        total_time = _all_reduce_sum(total_time, self.args)
        correct_end = _all_reduce_sum(correct_end, self.args)
        total_end = _all_reduce_sum(total_end, self.args)
        bad_time_tokens = _all_reduce_sum(bad_time_tokens, self.args)

        time_acc = float((correct_time.to(torch.float64) / total_time.clamp_min(1).to(torch.float64)).item())
        end_acc = float((correct_end.to(torch.float64) / total_end.clamp_min(1).to(torch.float64)).item())

        if _is_rank0(self.args):
            print(f"[gen-check] bad_time_tokens={int(bad_time_tokens.item())} (should be 0).")

        # -------------------------
        # sklearn metrics (distributed)
        # -------------------------
        if len(pred_buf) == 0:
            preds_local = torch.empty((0, int(pred_len), 1), device=self.device, dtype=torch.float32)
            trues_local = torch.empty((0, int(pred_len), 1), device=self.device, dtype=torch.float32)
        else:
            preds_local = torch.cat(pred_buf, dim=0)
            trues_local = torch.cat(true_buf, dim=0)

        mae, mse, rmse, _, r2, mae_16, mse_16, rmse_16, __16, r2_16,= _metric_distributed_sklearn(preds_local, trues_local)
        # print("################################################################################################")

        # print(preds_local.shape,trues_local.shape)
        # print(preds_local[:,:16].shape,trues_local[:,:16].shape)

        # print("################################################################################################")
        # mae_16, mse_16, rmse_16, _, r2_16 = _metric_distributed_sklearn(preds_local[:,:16], trues_local[:,:16])
        # print(mae_16, mse_16, rmse_16, r2_16)

        # back to train mode
        self.model.train()
        return time_acc, end_acc, mae, mse, rmse, r2, mae_16, mse_16, rmse_16, r2_16


    def _save_checkpoint_fsdp(
        self,
        ckpt_dir: str,
        optimizer: AdamW,
        scheduler,
        epoch: int,
        best_val_loss: Optional[float],
        meta: Dict[str, Any],
    ):
        if _is_rank0(self.args):
            os.makedirs(ckpt_dir, exist_ok=True)
        _barrier(self.args)

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
        )
        model_sd, optim_sd = get_state_dict(self.model, optimizer, options=options)

        if _is_rank0(self.args):
            payload = {
                "model": model_sd,
                "optimizer": optim_sd,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": int(epoch),
                "best_val_loss": best_val_loss,
                "args": vars(self.args),
                "meta": meta,
            }
            torch.save(payload, os.path.join(ckpt_dir, "checkpoint.pt"))

        _barrier(self.args)

    def _load_checkpoint_fsdp(self, ckpt_path: str, optimizer: AdamW, scheduler):
        payload = torch.load(ckpt_path, map_location="cpu")

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=False,
        )

        _barrier(self.args)
        set_state_dict(
            self.model,
            optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optimizer"],
            options=options,
        )
        _barrier(self.args)

        if scheduler is not None and payload.get("scheduler", None) is not None:
            scheduler.load_state_dict(payload["scheduler"])

        return payload

    @torch.inference_mode()
    def test(self, test_loader, raw_test_loader, meta: Dict[str, Any], epoch_tag: str = "Final"):
        Tout = int(meta["Tout"])
        pred_len = int(meta["pred_len"])

        test_loss, test_acc = self.eval_teacher_forcing(test_loader)

        if _is_rank0(self.args):
            msg = (
                f"[{epoch_tag}] "
                f"teacher_forcing test_loss={test_loss:.6f} test_acc={test_acc*100:.2f}%"
            )
            print(msg)
            with open(getattr(self.args, "result_file", "result.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n\n")
                f.flush()

        if _get_bool(self.args, "do_generate_eval", False):

            time_acc, end_acc, mae_s, mse_s, rmse_s, r2_s, mae_16, mse_16, rmse_16, r2_16 = self.eval_generate_all_distributed(
                raw_test_loader, Tout=Tout, pred_len=pred_len
            )
            if _is_rank0(self.args):
                msg = (
                    f"[{epoch_tag}] "
                    f"generate_time_token_acc={time_acc*100:.2f}% | "
                    f"generate_end_acc={end_acc*100:.2f}% | "
                    f"series mae={mae_s:.6f} mse={mse_s:.6f} rmse={rmse_s:.6f} r2={r2_s:.6f}"
                )

                msg_16 = (
                    f"series(step-avg-16) mae={mae_16:.6f} mse={mse_16:.6f} rmse={rmse_16:.6f} r2={r2_16:.6f}"
                )

                print(msg)
                with open(getattr(self.args, "result_file", "result.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n\n")
                    f.write(msg_16 + "\n\n")
                    f.flush()

    def train(self, setting: str):
        if _is_rank0(self.args):
            with open(getattr(self.args, "result_file", "result.txt"), "a", encoding="utf-8") as f:
                f.write(setting + "\n\n")
                f.flush()

        if getattr(self.args, "random_seed", None) is not None:
            _seed_everything(int(self.args.random_seed) + int(_rank(self.args)))

        um = _unwrap_fsdp(self.model)

        collate_fn = CollatorTS(
            tokenizer=self.tokenizer,
            ts_start_id=int(um.ts_start_id),
            ts_end_id=int(um.ts_end_id),
            base_vocab_size=int(um.base_vocab_size),
            codebook_size=int(self.args.codebook_size),
            pad_to_multiple_of=_get_int(self.args, "pad_to_multiple_of", 8),
            default_prompt_text=None,
            enable_prompt_cache=_get_bool(self.args, "enable_prompt_cache", True),
            prompt_cache_max_items=_get_int(self.args, "prompt_cache_max_items", 2048),
        )

        meta, train_loader, val_loader, test_loader, raw_test_loader = data_provider(self.args, collate_fn)

        Tin = int(meta.get("Tin", -1))
        Tout = int(meta["Tout"])
        pred_len = int(meta["pred_len"])

        if _is_rank0(self.args):
            print(f"[meta] Tin={Tin} Tout={Tout} pred_len={pred_len}")

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        if _is_rank0(self.args):
            os.makedirs(ckpt_dir, exist_ok=True)
        _barrier(self.args)

        early = EarlyStopping(
            patience=_get_int(self.args, "patience", 3),
            verbose=_is_rank0(self.args),
        )
        optimizer = self._select_optimizer()
        scheduler = self._build_scheduler(optimizer, len(train_loader))

        for epoch in range(1, _get_int(self.args, "train_epochs", 1) + 1):
            for ld in (train_loader, val_loader, test_loader):
                s = getattr(ld, "sampler", None)
                if hasattr(s, "set_epoch"):
                    s.set_epoch(epoch)

            tr_loss, tr_acc = self.train_one_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.eval_teacher_forcing(val_loader)
            test_loss, test_acc = self.eval_teacher_forcing(test_loader)

            if _is_rank0(self.args):
                msg = (
                    f"[Epoch {epoch}] "
                    f"train_loss={tr_loss:.6f} train_acc={tr_acc*100:.2f}% | "
                    f"val_loss={val_loss:.6f} val_acc={val_acc*100:.2f}% | "
                    f"test_loss={test_loss:.6f} test_acc={test_acc*100:.2f}%"
                )
                print(msg)
                with open(getattr(self.args, "result_file", "result.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n\n")
                    f.flush()

            improved_local = 0
            if _is_rank0(self.args):
                improved = early.step(val_loss)
                improved_local = 1 if improved else 0

            improved_t = torch.tensor([improved_local], device=self.device, dtype=torch.int64)
            improved_t = _all_reduce_sum(improved_t, self.args)
            improved_global = bool(int(improved_t.item()) > 0)

            if improved_global:
                self._save_checkpoint_fsdp(
                    ckpt_dir=ckpt_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=(early.best if _is_rank0(self.args) else None),
                    meta=meta,
                )

            stop_local = 1 if (_is_rank0(self.args) and early.early_stop) else 0
            stop_t = torch.tensor([stop_local], device=self.device, dtype=torch.int64)
            stop_t = _all_reduce_sum(stop_t, self.args)
            stop = int(stop_t.item() > 0)

            if stop:
                if _is_rank0(self.args):
                    print("Early stopping")
                break

            do_gen_each = _get_bool(self.args, "gen_eval_each_epoch", False)
            if do_gen_each:
                time_acc, end_acc, mae_s, mse_s, rmse_s, r2_s = self.eval_generate_all_distributed(
                    raw_test_loader, Tout=Tout, pred_len=pred_len
                )
                if _is_rank0(self.args):
                    msg = (
                        f"[Epoch {epoch}] "
                        f"generate_time_token_acc={time_acc*100:.2f}% | "
                        f"generate_end_acc={end_acc*100:.2f}% | "
                        f"series mae={mae_s:.6f} mse={mse_s:.6f} rmse={rmse_s:.6f} r2={r2_s:.6f}"
                    )
                    print(msg)
                    with open(getattr(self.args, "result_file", "result.txt"), "a", encoding="utf-8") as f:
                        f.write(msg + "\n\n")
                        f.flush()

            _barrier(self.args)

        best_path = os.path.join(ckpt_dir, "checkpoint.pt")
        if _is_rank0(self.args) and (not os.path.exists(best_path)):
            print(f"[warn] checkpoint not found: {best_path} (no improvement saved).")
        _barrier(self.args)

        if os.path.exists(best_path):
            self._load_checkpoint_fsdp(best_path, optimizer, scheduler)
            _barrier(self.args)
            if _is_rank0(self.args):
                print(f"Loaded best: {best_path}")

        self.test(test_loader=test_loader, raw_test_loader=raw_test_loader, meta=meta, epoch_tag="Final")
        return self.model

