import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LogitsProcessorList, LogitsProcessor
)
import random
from peft import get_peft_model, LoraConfig, TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from .sparselinear import SparseLinear
import math


from typing import Optional, Tuple, Dict, Any

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 4 * d_model

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)

        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x, context, attn_mask=None, key_padding_mask=None):
        """
        x:      [B, Lq, D]
        context:[B, Lk, D]
        """

        residual = x
        attn_out, _ = self.attn(
            query=x,
            key=context,
            value=context,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(residual + self.drop1(attn_out))

        residual = x
        x = self.fc2(self.drop2(self.act(self.fc1(x))))
        x = self.norm2(residual + x)
        return x
    
class Qwen3RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., normalized_shape)
        # rms = sqrt(mean(x^2) + eps)
        # output = x / rms * weight
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean_sq = x.pow(2).mean(dim=dims, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return x_normed * self.weight


# ----------------------------
# Rotary Embedding (RoPE)
# ----------------------------
class Qwen3RotaryEmbedding(nn.Module):
    """
    Standard RoPE for head_dim (must be even).
    Caches cos/sin for efficiency.
    """
    def __init__(self, dim: int = 128, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even"
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0
        self._device_cached: Optional[torch.device] = None
        self._dtype_cached: Optional[torch.dtype] = None

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # positions: (seq_len,)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        # Interleave to (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len
        self._device_cached = device
        self._dtype_cached = dtype

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._device_cached != device
            or self._dtype_cached != dtype
        ):
            self._build_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (B, Hq, T, D)
        k: (B, Hk, T, D)
        position_ids: (B, T) or (T,) optional. If None, assumes positions [0..T-1].
        """
        B, Hq, T, D = q.shape
        _, Hk, Tk, Dk = k.shape
        assert T == Tk and D == Dk == self.dim

        device = q.device
        dtype = q.dtype

        if cos is None or sin is None:
            cos_full, sin_full = self.get_cos_sin(T, device=device, dtype=dtype)  # (T, D)
        else:
            cos_full, sin_full = cos, sin

        if position_ids is None:
            # (T, D) -> (1, 1, T, D)
            cos_t = cos_full.view(1, 1, T, D)
            sin_t = sin_full.view(1, 1, T, D)
        else:
            # position_ids: (B, T) or (T,)
            if position_ids.dim() == 1:
                pos = position_ids  # (T,)
                cos_t = cos_full.index_select(0, pos).view(1, 1, T, D)
                sin_t = sin_full.index_select(0, pos).view(1, 1, T, D)
            else:
                # (B, T) -> gather (B, T, D) -> (B, 1, T, D)
                cos_bt = cos_full.index_select(0, position_ids.reshape(-1)).view(B, T, D)
                sin_bt = sin_full.index_select(0, position_ids.reshape(-1)).view(B, T, D)
                cos_t = cos_bt.unsqueeze(1)
                sin_t = sin_bt.unsqueeze(1)

        q_out = (q * cos_t) + (self._rotate_half(q) * sin_t)
        k_out = (k * cos_t) + (self._rotate_half(k) * sin_t)
        return q_out, k_out


# ----------------------------
# MLP (SwiGLU-like gating)
# ----------------------------
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-ish: SiLU(gate(x)) * up(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ----------------------------
# Attention (GQA)
# ----------------------------
def _make_causal_mask(t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # (1, 1, T, T)
    mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, t, t)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        q_out: int = 2048,
        kv_out: int = 1024,
        head_dim: int = 128,
        eps: float = 1e-6,
        rotary_emb: Optional[Qwen3RotaryEmbedding] = None,
    ):
        super().__init__()
        assert q_out % head_dim == 0
        assert kv_out % head_dim == 0

        self.hidden_size = hidden_size
        self.q_out = q_out
        self.kv_out = kv_out
        self.head_dim = head_dim

        self.n_q_heads = q_out // head_dim     # 2048/128 = 16
        self.n_kv_heads = kv_out // head_dim   # 1024/128 = 8

        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=False)

        # head-wise RMSNorm for q/k with shape (head_dim,)
        self.q_norm = Qwen3RMSNorm((head_dim,), eps=eps)
        self.k_norm = Qwen3RMSNorm((head_dim,), eps=eps)

        self.rotary_emb = rotary_emb if rotary_emb is not None else Qwen3RotaryEmbedding(dim=head_dim)

    def _shape_q(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        # (B, T, 2048) -> (B, Hq, T, D)
        return x.view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_kv(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        # (B, T, 1024) -> (B, Hk, T, D)
        return x.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        # kv: (B, Hk, T, D) -> (B, Hq, T, D) by repeating along head axis
        if self.n_q_heads == self.n_kv_heads:
            return kv
        assert self.n_q_heads % self.n_kv_heads == 0
        rep = self.n_q_heads // self.n_kv_heads
        return kv.repeat_interleave(rep, dim=1)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, 1024)
        attention_mask: Optional[torch.Tensor] = None,  # additive mask broadcastable to (B, 1, T, S)
        position_ids: Optional[torch.Tensor] = None,    # (B, T) or (T,)
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (k, v) each (B, Hk, S_past, D)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape

        q = self._shape_q(self.q_proj(x), b, t)     # (B, Hq, T, D)
        k = self._shape_kv(self.k_proj(x), b, t)    # (B, Hk, T, D)
        v = self._shape_kv(self.v_proj(x), b, t)    # (B, Hk, T, D)

        # head-wise norm on last dim
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE applied on current sequence positions
        q, k = self.rotary_emb.apply_rope(q, k, position_ids=position_ids)

        # append cache if any
        if past_kv is not None:
            past_k, past_v = past_kv
            # concat on sequence dim
            k = torch.cat([past_k, k], dim=2)  # (B, Hk, S, D)
            v = torch.cat([past_v, v], dim=2)  # (B, Hk, S, D)

        present = (k, v) if use_cache else None

        # GQA: expand k/v heads to match q heads for attention computation
        k_for_attn = self._repeat_kv(k)  # (B, Hq, S, D)
        v_for_attn = self._repeat_kv(v)  # (B, Hq, S, D)

        s = k_for_attn.size(2)  # total source length

        # scaled dot-product attention with causal mask
        # scores: (B, Hq, T, S)
        scores = torch.matmul(q, k_for_attn.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # causal: each token can only attend to <= its position in the *combined* (past+current) sequence
        # when cache exists, causal mask should allow attending to all past, but still prevent attending to future in current.
        # build causal mask for (T, S)
        # If S == T (no past): standard triangular.
        # If S > T: we need mask that blocks attention to positions > (past_len + i) for each i in [0..T-1]
        if s == t:
            causal = _make_causal_mask(t, device=x.device, dtype=scores.dtype)  # (1,1,T,T)
        else:
            past_len = s - t
            # create (T, S) mask with -inf where j > past_len + i
            i = torch.arange(t, device=x.device).unsqueeze(1)          # (T,1)
            j = torch.arange(s, device=x.device).unsqueeze(0)          # (1,S)
            bad = j > (past_len + i)
            causal = torch.zeros((t, s), device=x.device, dtype=scores.dtype)
            causal = causal.masked_fill(bad, float("-inf")).view(1, 1, t, s)

        scores = scores + causal

        if attention_mask is not None:
            # additive mask (e.g. padding) broadcastable to (B, 1, T, S)
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_for_attn)  # (B, Hq, T, D)

        # merge heads: (B, T, Hq*D=2048)
        out = out.transpose(1, 2).contiguous().view(b, t, self.q_out)
        out = self.o_proj(out)  # (B, T, 1024)
        return out, present


# ----------------------------
# Decoder Layer
# ----------------------------
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        head_dim: int = 128,
        eps: float = 1e-6,
        rotary_emb: Optional[Qwen3RotaryEmbedding] = None,
    ):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            q_out=2048,
            kv_out=1024,
            head_dim=head_dim,
            eps=eps,
            rotary_emb=rotary_emb,
        )
        self.mlp = Qwen3MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = Qwen3RMSNorm((hidden_size,), eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm((hidden_size,), eps=eps)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, 1024)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm
        h = self.input_layernorm(x)
        attn_out, present = self.self_attn(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out

        h2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(h2)
        x = x + mlp_out
        return x, present


# ----------------------------
# Full Decoder stack (as in your print)
# ----------------------------
class Qwen3Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        # share one rotary embedding module across layers (common practice)
        self.rotary_emb = Qwen3RotaryEmbedding(dim=head_dim)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                eps=eps,
                rotary_emb=self.rotary_emb,
            )
            for _ in range(num_layers)
        ])
        self.norm = Qwen3RMSNorm((hidden_size,), eps=eps)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, 1024)
        attention_mask: Optional[torch.Tensor] = None,  # additive
        position_ids: Optional[torch.Tensor] = None,
        past_kvs: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_kv=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        x = self.norm(x)
        # if use_cache:
        #     return x, tuple(presents)
        return x

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        config = AutoConfig.from_pretrained(configs.LLM_model_path)
        self.d_model = config.hidden_size
        num_heads = config.num_attention_heads
        # dim_ff = 2048
        # dropout = 0.1
        # num_layers = 38
        # codebook_size = 256

        self.embed_tokens = nn.Embedding(151936+configs.codebook_size, 1024)

        self.encoder = Qwen3Decoder()



        self.Fusion =CrossAttentionBlock(d_model = 1024, num_heads = num_heads)

        self.ts_head = SparseLinear(1024, configs.codebook_size)
        
    def forward(self, inputs): 
        x = self.embed_tokens(inputs)   
        outputs = self.encoder(x)

        hidden = outputs
        hidden = self.Fusion(hidden[:,-26:,:], hidden[:,:-26,:])
        hidden_ts = hidden[:,-25:-1,:]
        ts_logits = self.ts_head(hidden_ts)
        outputs.logits = ts_logits
    
        return outputs


# import torch
# import torch.nn as nn
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     AutoConfig,
#     LogitsProcessorList, LogitsProcessor
# )
# import random
# from peft import get_peft_model, LoraConfig, TaskType

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from peft import LoraConfig, get_peft_model
# from .sparselinear import SparseLinear
# import math

# def find_all_linear_names(model: nn.Module):
#     """Find module name suffixes for all nn.Linear modules (LoRA target_modules)."""
#     cls = nn.Linear
#     names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             parts = name.split(".")
#             names.add(parts[-1])
#     # Common: lm_head shouldn't be LoRA'd in many setups
#     names.discard("lm_head")
#     return list(names)


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
#         d_ff = 4 * d_model

#         self.attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.drop1 = nn.Dropout(dropout)

#         self.norm2 = nn.LayerNorm(d_model)
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.drop2 = nn.Dropout(dropout)

#         self.act = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, context, attn_mask=None, key_padding_mask=None):
#         """
#         x:      [B, Lq, D]
#         context:[B, Lk, D]
#         """

#         residual = x
#         attn_out, _ = self.attn(
#             query=x,
#             key=context,
#             value=context,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )
#         x = self.norm1(residual + self.drop1(attn_out))

#         residual = x
#         x = self.fc2(self.drop2(self.act(self.fc1(x))))
#         x = self.norm2(residual + x)
#         return x

# class Model(nn.Module):
#     def __init__(self,configs):
#         super(Model, self).__init__()
#         config = AutoConfig.from_pretrained(configs.LLM_model_path)
#         self.d_model = config.hidden_size
#         self.text_tokenizer = AutoTokenizer.from_pretrained(configs.LLM_model_path)
#         # self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
#         # 添加时序特殊token
#         special_tokens_dict = {
#         'additional_special_tokens': ['<TS_START>', '<TS_END>']
#         }

#         self.text_tokenizer.add_special_tokens(special_tokens_dict)
#         # self.text_tokenizer.apply_chat_template

#         # 初始化模型
#         self.model = AutoModelForCausalLM.from_pretrained(
#                 configs.LLM_model_path,
#                 # torch_dtype=torch.bfloat16, 
#                 output_attentions=True, 
#                 output_hidden_states=True,
#                 trust_remote_code=True
#             )
#         print(self.model)
#         self.model.config.use_cache = False
#         ori_vocabe_size = len(self.model.model.embed_tokens.weight)
#         self.ori_vocabe_size = ori_vocabe_size

#         self.model.resize_token_embeddings(new_num_tokens=ori_vocabe_size+configs.codebook_size)
    
#         # 然后随机初始化模型权重
#         self.model.init_weights()  # 重新初始化所有权重

#         self.Fusion =CrossAttentionBlock(d_model = self.model.model.embed_tokens.weight.shape[-1], num_heads = config.num_attention_heads).to(dtype=torch.bfloat16)

#         self.ts_head = SparseLinear(self.model.config.hidden_size, configs.codebook_size).to(dtype=torch.bfloat16)
        
#     def forward(self, inputs):    
   
#         outputs = self.model(
#             input_ids=inputs,
#             output_hidden_states=True
#         )
#         hidden = outputs.hidden_states[-1]
#         hidden = self.Fusion(hidden[:,-26:,:], hidden[:,:-26,:])
#         hidden_ts = hidden[:,-25:-1,:]
#         ts_logits = self.ts_head(hidden_ts)
#         outputs.logits = ts_logits
    
#         return outputs


