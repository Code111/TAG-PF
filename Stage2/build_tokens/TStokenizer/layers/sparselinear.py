# import math
# import warnings
# import numpy as np
# import torch
# import torch.nn as nn


# def small_world_chunker(inputs, outputs, nnz):
#     """Utility function for small world initialization as presented in the write up Bipartite_small_world_network"""
#     pair_distance = inputs.view(-1, 1) - outputs
#     arg = torch.abs(pair_distance) + 1.0

#     # lambda search
#     L, U = 1e-5, 5.0
#     lamb = 1.0  # initial guess
#     itr = 1
#     error_threshold = 10.0
#     max_itr = 1000
#     P = arg ** (-lamb)
#     P_sum = P.sum()
#     error = abs(P_sum - nnz)

#     while error > error_threshold:
#         assert itr <= max_itr, "No solution found; please try different network sizes and sparsity levels"
#         if P_sum < nnz:
#             U = lamb
#             lamb = (lamb + L) / 2.0
#         else:
#             L = lamb
#             lamb = (lamb + U) / 2.0

#         P = arg ** (-lamb)
#         P_sum = P.sum()
#         error = abs(P_sum - nnz)
#         itr += 1
#     return P


# def _coo(indices, values, size, ref_tensor=None):
#     """
#     统一构造稀疏 COO tensor，保证 dtype/device 正确，并 coalesce。
#     """
#     if ref_tensor is not None:
#         device = ref_tensor.device
#         dtype = ref_tensor.dtype
#     else:
#         device = values.device
#         dtype = values.dtype

#     return torch.sparse_coo_tensor(
#         indices=indices,
#         values=values,
#         size=size,
#         device=device,
#         dtype=dtype
#     ).coalesce()


# class GrowConnections(torch.autograd.Function):
#     """ Custom pytorch function to handle growing connections """

#     @staticmethod
#     def forward(ctx, inputs, weights, k, indices, features, max_size):
#         out_features, in_features = features
#         output_shape = list(inputs.shape)
#         output_shape[-1] = out_features

#         if len(output_shape) == 1:
#             inputs = inputs.view(1, -1)
#         inputs = inputs.flatten(end_dim=-2)  # [N, in_features]

#         # ✅ 使用真正 sparse mm（不 to_dense）
#         W = _coo(indices.long(), weights, (out_features, in_features), ref_tensor=inputs)
#         output = torch.sparse.mm(W, inputs.t()).t()  # [N, out_features]

#         ctx.save_for_backward(inputs, weights, indices)
#         ctx.k = k
#         ctx.out_features = out_features
#         ctx.in_features = in_features
#         ctx.max_size = max_size

#         return output.view(output_shape)

#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, weights, indices = ctx.saved_tensors
#         k = ctx.k
#         out_features = ctx.out_features
#         in_features = ctx.in_features
#         max_size = ctx.max_size

#         # grad_output 形状对齐
#         if grad_output.dim() == 1:
#             grad_output = grad_output.view(1, -1)
#         grad_output = grad_output.flatten(end_dim=-2)  # [N, out_features]

#         # ✅ grad_input = grad_output @ W
#         # W: [out, in]  -> W^T: [in, out]
#         W_t = _coo(indices[[1, 0]].long(), weights, (in_features, out_features), ref_tensor=grad_output)
#         grad_input = torch.sparse.mm(W_t, grad_output.t()).t()  # [N, in_features]

#         # -------------------------------
#         # Growth criterion (挑选新的连接)
#         # -------------------------------
#         if in_features * out_features <= max_size:
#             # grad_weights_full: [out, in]
#             grad_weights_full = torch.matmul(grad_output.t(), inputs)  # [out_features, in_features]
#             grad_weights_full = torch.abs(grad_weights_full)

#             mask = torch.ones_like(grad_weights_full)
#             mask[indices[0], indices[1]] = 0
#             masked_weights = mask * grad_weights_full

#             _, lm_indices = torch.topk(masked_weights.reshape(-1), k, sorted=False)
#             row = lm_indices.floor_divide(in_features)
#             col = lm_indices.fmod(in_features)
#         else:
#             # chunk top-k（保持你原逻辑，但修正类型）
#             tk = None
#             m = max_size / in_features
#             chunks = math.ceil(out_features / m)

#             for item in range(chunks):
#                 start = int(item * m)
#                 end = int((item + 1) * m) if item != chunks - 1 else out_features

#                 # grad_m: [end-start, in]
#                 grad_m = torch.matmul(grad_output[:, start:end].t(), inputs)  # [block_out, in]
#                 grad_m_abs = torch.abs(grad_m)

#                 topk_values, topk_indices = torch.topk(grad_m_abs.view(-1), k, sorted=False)

#                 row_local = topk_indices.floor_divide(in_features) + start
#                 col_local = topk_indices.fmod(in_features)

#                 cur = torch.stack([topk_values, row_local.to(topk_values.dtype), col_local.to(topk_values.dtype)], dim=0)

#                 if tk is None:
#                     tk = cur
#                 else:
#                     concat_vals = torch.cat([tk[0], cur[0]], dim=0)
#                     topk_vals2, topk_idx2 = torch.topk(concat_vals, k, sorted=False)

#                     # 取出对应 row/col
#                     all_rows = torch.cat([tk[1], cur[1]], dim=0)
#                     all_cols = torch.cat([tk[2], cur[2]], dim=0)
#                     tk = torch.stack([topk_vals2, all_rows[topk_idx2], all_cols[topk_idx2]], dim=0)

#             row = tk[1].long()
#             col = tk[2].long()

#         # 将新增连接写回 indices（保持你原 “替换最后 k 个连接” 的策略）
#         new_indices = torch.stack((row, col))
#         x = torch.cat((indices[:, :-k], new_indices), dim=1)

#         # 若长度不一致，补零（你原逻辑保留）
#         if indices.shape[1] > x.shape[1]:
#             diff = indices.shape[1] - x.shape[1]
#             new_entries = torch.zeros((2, diff), dtype=torch.long, device=x.device)
#             x = torch.cat((x, new_entries), dim=1)

#         indices.copy_(x)

#         # 注意：这里只返回 grad_input 和 weights 的梯度（其它 None）
#         # weights 的梯度由 autograd 自动算不了，因为 indices 会变；这里保持原设计：不回传 grad_weights
#         return grad_input.view_as(inputs), None, None, None, None, None


# class SparseLinear(nn.Module):
#     """
#     Sparse Linear layer with optional small-world init and dynamic sparsity.

#     Input:  (N, *, in_features)
#     Output: (N, *, out_features)
#     """

#     def __init__(
#         self,
#         in_features,
#         out_features,
#         bias=True,
#         sparsity=0.9,
#         connectivity=None,
#         small_world=False,
#         dynamic=False,
#         deltaT=6000,
#         Tend=150000,
#         alpha=0.1,
#         max_size=1e8,
#     ):
#         super().__init__()

#         assert in_features < 2**31 and out_features < 2**31 and sparsity < 1.0
#         assert connectivity is None or not small_world, "Cannot specify connectivity along with small world sparsity"

#         self.in_features = in_features
#         self.out_features = out_features
#         self.connectivity = connectivity
#         self.small_world = small_world
#         self.dynamic = dynamic
#         self.max_size = max_size

#         # indices 生成位置（cpu/gpu 都行）
#         coalesce_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#         if not small_world:
#             if connectivity is None:
#                 self.sparsity = sparsity
#                 nnz = round((1.0 - sparsity) * in_features * out_features)

#                 if in_features * out_features <= 10**8:
#                     idx = np.random.choice(in_features * out_features, nnz, replace=False)
#                     idx = torch.as_tensor(idx, device=coalesce_device)
#                     row_ind = idx.floor_divide(in_features)
#                     col_ind = idx.fmod(in_features)
#                 else:
#                     warnings.warn(
#                         "Matrix too large to sample non-zero indices without replacement, sparsity will be approximate",
#                         RuntimeWarning,
#                     )
#                     row_ind = torch.randint(0, out_features, (nnz,), device=coalesce_device)
#                     col_ind = torch.randint(0, in_features, (nnz,), device=coalesce_device)

#                 indices = torch.stack((row_ind, col_ind))
#             else:
#                 nnz = connectivity.shape[1]
#                 self.sparsity = 1.0 - nnz / (out_features * in_features)
#                 indices = connectivity.to(device=coalesce_device)
#         else:
#             self.sparsity = sparsity
#             nnz = round((1.0 - sparsity) * in_features * out_features)
#             assert nnz > min(in_features, out_features), "Too sparse for small-world; please decrease sparsity"
#             offset = abs(out_features - in_features) / 2.0

#             inputs = torch.arange(
#                 1 + offset * (out_features > in_features),
#                 in_features + 1 + offset * (out_features > in_features),
#                 device=coalesce_device,
#             )
#             outputs = torch.arange(
#                 1 + offset * (out_features < in_features),
#                 out_features + 1 + offset * (out_features < in_features),
#                 device=coalesce_device,
#             )

#             total_data = in_features * out_features
#             chunks = math.ceil(total_data / self.max_size)
#             split_div = max(in_features, out_features) // chunks
#             split_mod = max(in_features, out_features) % chunks

#             idx = torch.repeat_interleave(torch.tensor([split_div], device=coalesce_device), chunks).int()
#             idx[:split_mod] += 1
#             idx = torch.cumsum(idx, dim=0)
#             idx = torch.cat([torch.LongTensor([0]).to(device=coalesce_device), idx])

#             rows = torch.empty(0, dtype=torch.long, device=coalesce_device)
#             cols = torch.empty(0, dtype=torch.long, device=coalesce_device)

#             for i in range(chunks):
#                 inputs_ = inputs[idx[i]:idx[i + 1]] if out_features <= in_features else inputs
#                 outputs_ = outputs[idx[i]:idx[i + 1]] if out_features > in_features else outputs

#                 y = small_world_chunker(inputs_, outputs_, round(nnz / chunks))
#                 ref = torch.rand_like(y)

#                 mask = (y >= ref)
#                 rows_, cols_ = mask.to_sparse().indices()

#                 rows = torch.cat([rows, rows_ + idx[i]])
#                 cols = torch.cat([cols, cols_])

#             indices = torch.stack((cols, rows))
#             nnz = indices.shape[1]

#         # 参数
#         values = torch.empty(nnz, device=coalesce_device)

#         self.register_buffer("indices", indices.long())
#         self.weights = nn.Parameter(values)

#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter("bias", None)

#         if self.dynamic:
#             self.deltaT = deltaT
#             self.Tend = Tend
#             self.alpha = alpha
#             self.itr_count = 0

#         self.reset_parameters()

#     def reset_parameters(self):
#         bound = 1 / (self.in_features ** 0.5)
#         nn.init.uniform_(self.weights, -bound, bound)
#         if self.bias is not None:
#             nn.init.uniform_(self.bias, -bound, bound)

#     @property
#     def weight(self):
#         """Return sparse COO weight (for inspection only)."""
#         W = _coo(self.indices, self.weights, (self.out_features, self.in_features), ref_tensor=self.weights)
#         return W.detach()

#     def forward(self, inputs):
#         if self.training and self.dynamic:
#             self.itr_count += 1

#         output_shape = list(inputs.shape)
#         output_shape[-1] = self.out_features

#         if len(output_shape) == 1:
#             inputs = inputs.view(1, -1)

#         flat_in = inputs.flatten(end_dim=-2)  # [N, in_features]

#         # ---------------------------
#         # 动态稀疏：周期性 prune/grow
#         # ---------------------------
#         if (
#             self.training
#             and self.dynamic
#             and self.itr_count < self.Tend
#             and self.itr_count % self.deltaT == 0
#         ):
#             f_decay = self.alpha * (1 + math.cos(self.itr_count * math.pi / self.Tend)) / 2
#             k = int(f_decay * (1 - self.sparsity) * self.weights.numel())
#             n = self.weights.numel()

#             # prune：保留绝对值大的（删掉最小的 n-k）
#             neg_weights = -torch.abs(self.weights)
#             _, lm_indices = torch.topk(neg_weights, n - k, largest=False, sorted=False)

#             self.indices = torch.index_select(self.indices, 1, lm_indices)
#             self.weights = nn.Parameter(torch.index_select(self.weights, 0, lm_indices))

#             # grow：补 k 个新连接（由 GrowConnections.backward 决定位置）
#             new_weights = torch.zeros(k, device=flat_in.device, dtype=flat_in.dtype)
#             self.weights = nn.Parameter(torch.cat((self.weights, new_weights), dim=0))

#             new_indices = torch.zeros((2, k), dtype=torch.long, device=flat_in.device)
#             self.indices = torch.cat((self.indices.to(flat_in.device), new_indices), dim=1)

#             out = GrowConnections.apply(
#                 flat_in,
#                 self.weights,
#                 k,
#                 self.indices,
#                 (self.out_features, self.in_features),
#                 self.max_size,
#             )
#         else:
#             # ✅ 正常 sparse mm
#             W = _coo(self.indices.to(flat_in.device), self.weights, (self.out_features, self.in_features), ref_tensor=flat_in)
#             out = torch.sparse.mm(W, flat_in.t()).t()  # [N, out_features]

#         if self.bias is not None:
#             out = out + self.bias.to(out.device, out.dtype)

#         return out.view(output_shape)

#     def extra_repr(self):
#         return (
#             f"in_features={self.in_features}, out_features={self.out_features}, "
#             f"bias={self.bias is not None}, sparsity={self.sparsity:.4f}, "
#             f"connectivity={'None' if self.connectivity is None else 'Provided'}, small_world={self.small_world}, dynamic={self.dynamic}"
#         )

import math
import warnings
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Small-world 初始化辅助
# -------------------------
def small_world_chunker(inputs, outputs, nnz):
    """Small-world bipartite init: probability proportional to |i-j|^{-lambda}."""
    pair_distance = inputs.view(-1, 1) - outputs
    arg = torch.abs(pair_distance) + 1.0

    # lambda search
    L, U = 1e-5, 5.0
    lamb = 1.0
    itr = 1
    error_threshold = 10.0
    max_itr = 1000

    P = arg ** (-lamb)
    P_sum = P.sum()
    error = abs(P_sum - nnz)

    while error > error_threshold:
        if itr > max_itr:
            raise RuntimeError("No solution found; try different sizes/sparsity")
        if P_sum < nnz:
            U = lamb
            lamb = (lamb + L) / 2.0
        else:
            L = lamb
            lamb = (lamb + U) / 2.0

        P = arg ** (-lamb)
        P_sum = P.sum()
        error = abs(P_sum - nnz)
        itr += 1

    return P


def _coo(indices, values, size, ref_tensor=None):
    """构造 sparse COO，保证 device/dtype，并 coalesce。"""
    if ref_tensor is not None:
        device = ref_tensor.device
        dtype = ref_tensor.dtype
    else:
        device = values.device
        dtype = values.dtype

    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        device=device,
        dtype=dtype,
    ).coalesce()


# -------------------------
# 动态稀疏 Autograd Kernel
# -------------------------
class _DynamicSparseMM(torch.autograd.Function):
    """
    forward: sparse.mm
    backward:
      - grad_input
      - grad_weights（只对 active 边）
      - 如果 grow_slots 非空：用 |grad_out^T @ inp| 选新边，写回 indices/active_mask
    """

    @staticmethod
    def forward(
        ctx,
        inputs,           # [N, *, in]
        weights,          # [M] Parameter (fixed)
        indices,          # [2, M] buffer
        active_mask,      # [M] bool buffer
        grow_slots,       # [k] long (slots pruned this step; to be regrown)
        out_features: int,
        in_features: int,
        max_size: int,
    ):
        if inputs.dim() == 1:
            inputs = inputs.view(1, -1)

        x = inputs.flatten(end_dim=-2)  # [N, in]
        active_pos = torch.nonzero(active_mask, as_tuple=False).squeeze(1)

        idx_act = indices.index_select(1, active_pos).long()       # [2, nnz_act]
        w_act = weights.index_select(0, active_pos)                # [nnz_act]

        W = _coo(idx_act, w_act, (out_features, in_features), ref_tensor=x)
        y = torch.sparse.mm(W, x.t()).t()  # [N, out]

        ctx.save_for_backward(x, weights, indices, active_mask, active_pos, grow_slots)
        ctx.out_features = int(out_features)
        ctx.in_features = int(in_features)
        ctx.max_size = int(max_size)

        return y.view(*inputs.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        x, weights, indices, active_mask, active_pos, grow_slots = ctx.saved_tensors
        out_features = ctx.out_features
        in_features = ctx.in_features
        max_size = ctx.max_size

        if grad_output.dim() == 1:
            grad_output = grad_output.view(1, -1)
        g = grad_output.flatten(end_dim=-2)  # [N, out]

        # active edges
        idx_act = indices.index_select(1, active_pos).long()
        row = idx_act[0]  # [nnz_act]
        col = idx_act[1]  # [nnz_act]
        w_act = weights.index_select(0, active_pos)

        # ---- grad_input = g @ W ----
        Wt = _coo(torch.stack([col, row], dim=0), w_act, (in_features, out_features), ref_tensor=g)
        grad_x = torch.sparse.mm(Wt, g.t()).t()  # [N, in]
        grad_input = grad_x.view(*grad_output.shape[:-1], in_features)

        # ---- grad_weights for active edges ----
        # grad_w[e] = sum_n g[n,row[e]] * x[n,col[e]]
        x_sel = x[:, col]          # [N, nnz_act]
        g_sel = g[:, row]          # [N, nnz_act]
        grad_w_act = (x_sel * g_sel).sum(dim=0)  # [nnz_act]

        grad_weights = torch.zeros_like(weights)
        grad_weights.scatter_add_(0, active_pos, grad_w_act)

        # ---- grow: fill grow_slots with new edges (in-place) ----
        if grow_slots is not None and grow_slots.numel() > 0:
            k = int(grow_slots.numel())

            # existing edges linear index
            lin_exist = (row * in_features + col).long()

            if out_features * in_features <= max_size:
                # dense score is allowed
                scores = torch.matmul(g.t(), x).abs()  # [out, in]
                flat = scores.reshape(-1)

                # mask out existing edges
                mask = torch.ones(out_features * in_features, device=flat.device, dtype=torch.bool)
                mask[lin_exist] = False
                flat = flat.masked_fill(~mask, 0)

                _, topi = torch.topk(flat, k, sorted=False)
                new_row = topi.floor_divide(in_features).long()
                new_col = topi.fmod(in_features).long()
            else:
                # chunked top-k: reduce peak memory
                tk_vals = None
                tk_row = None
                tk_col = None
                tk_lin = None

                m = max_size / in_features
                chunks = math.ceil(out_features / m)

                for item in range(chunks):
                    start = int(item * m)
                    end = int((item + 1) * m) if item != chunks - 1 else out_features

                    block = torch.matmul(g[:, start:end].t(), x).abs()  # [block_out, in]
                    flatb = block.reshape(-1)

                    vals, idx = torch.topk(flatb, k, sorted=False)
                    rb = idx.floor_divide(in_features).long() + start
                    cb = idx.fmod(in_features).long()
                    linb = (rb * in_features + cb).long()

                    if tk_vals is None:
                        tk_vals, tk_row, tk_col, tk_lin = vals, rb, cb, linb
                    else:
                        all_vals = torch.cat([tk_vals, vals], dim=0)
                        all_row = torch.cat([tk_row, rb], dim=0)
                        all_col = torch.cat([tk_col, cb], dim=0)
                        all_lin = torch.cat([tk_lin, linb], dim=0)

                        v2, p2 = torch.topk(all_vals, k, sorted=False)
                        tk_vals = v2
                        tk_row = all_row[p2]
                        tk_col = all_col[p2]
                        tk_lin = all_lin[p2]

                # remove existing edges; if insufficient, allow duplicates to avoid crash
                keep = ~torch.isin(tk_lin, lin_exist)
                new_row = tk_row[keep]
                new_col = tk_col[keep]
                if new_row.numel() < k:
                    need = k - new_row.numel()
                    new_row = torch.cat([new_row, tk_row[:need]], dim=0)
                    new_col = torch.cat([new_col, tk_col[:need]], dim=0)
                else:
                    new_row = new_row[:k]
                    new_col = new_col[:k]

            # in-place write back
            indices[0, grow_slots] = new_row.to(indices.device)
            indices[1, grow_slots] = new_col.to(indices.device)
            active_mask[grow_slots] = True

        # Return gradients for (inputs, weights, indices, active_mask, grow_slots, out_f, in_f, max_size)
        return grad_input, grad_weights, None, None, None, None, None, None


# -------------------------
# 完整 SparseLinear 模块
# -------------------------
class SparseLinear(nn.Module):
    """
    Sparse Linear layer with:
      - (optional) small-world init
      - (optional) connectivity init
      - (optional) dynamic sparsity (prune + grow)
    IMPORTANT:
      - weights is a fixed-size nn.Parameter -> optimizer never loses track.
      - indices/active_mask are buffers updated in-place.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9,
        connectivity=None,         # torch.LongTensor [2, nnz]
        small_world: bool = False,
        dynamic: bool = False,
        deltaT: int = 6000,
        Tend: int = 150000,
        alpha: float = 0.1,
        max_size: int = int(1e8),
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float32

        assert 0.0 <= sparsity < 1.0
        assert in_features < 2**31 and out_features < 2**31
        assert connectivity is None or not small_world, "Cannot use connectivity with small_world"

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.small_world = bool(small_world)
        self.dynamic = bool(dynamic)
        self.sparsity = float(sparsity)
        self.max_size = int(max_size)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # ---- build initial indices ----
        if connectivity is not None:
            indices = connectivity.to(device=device).long()
            nnz = int(indices.shape[1])
            self.sparsity = 1.0 - nnz / (out_features * in_features)
        else:
            nnz = round((1.0 - sparsity) * in_features * out_features)
            nnz = int(nnz)

            if not small_world:
                if in_features * out_features <= 10**8:
                    idx = np.random.choice(in_features * out_features, nnz, replace=False)
                    idx = torch.as_tensor(idx, device=device)
                    row = idx // in_features
                    col = idx % in_features
                else:
                    warnings.warn(
                        "Matrix too large to sample without replacement; sparsity will be approximate",
                        RuntimeWarning,
                    )
                    row = torch.randint(0, out_features, (nnz,), device=device)
                    col = torch.randint(0, in_features, (nnz,), device=device)
                indices = torch.stack((row, col), dim=0).long()
            else:
                # small-world init (chunked to avoid huge allocation)
                assert nnz > min(in_features, out_features), "Too sparse for small-world; decrease sparsity"

                offset = abs(out_features - in_features) / 2.0
                inputs = torch.arange(
                    1 + offset * (out_features > in_features),
                    in_features + 1 + offset * (out_features > in_features),
                    device=device,
                )
                outputs = torch.arange(
                    1 + offset * (out_features < in_features),
                    out_features + 1 + offset * (out_features < in_features),
                    device=device,
                )

                total = in_features * out_features
                chunks = math.ceil(total / self.max_size)
                split_div = max(in_features, out_features) // chunks
                split_mod = max(in_features, out_features) % chunks

                idxs = torch.repeat_interleave(torch.tensor([split_div], device=device), chunks).int()
                idxs[:split_mod] += 1
                idxs = torch.cumsum(idxs, dim=0)
                idxs = torch.cat([torch.tensor([0], device=device, dtype=torch.long), idxs.long()])

                rows = torch.empty(0, dtype=torch.long, device=device)
                cols = torch.empty(0, dtype=torch.long, device=device)

                for i in range(chunks):
                    inputs_ = inputs[idxs[i]:idxs[i + 1]] if out_features <= in_features else inputs
                    outputs_ = outputs[idxs[i]:idxs[i + 1]] if out_features > in_features else outputs

                    P = small_world_chunker(inputs_, outputs_, round(nnz / chunks))
                    ref = torch.rand_like(P)
                    mask = (P >= ref)
                    r_, c_ = mask.to_sparse().indices()
                    rows = torch.cat([rows, r_ + idxs[i]])
                    cols = torch.cat([cols, c_])

                # 注意 small_world 这块你的原实现里有 cols/rows 交换，这里保持一致
                indices = torch.stack((cols, rows), dim=0).long()
                nnz = int(indices.shape[1])

        self.nnz_max = int(nnz)

        # ---- fixed parameter weights ----
        self.weights = nn.Parameter(torch.empty(self.nnz_max, device=device, dtype=dtype))

        # ---- buffers (must stay buffers) ----
        self.register_buffer("indices", indices.clone())
        self.register_buffer("active_mask", torch.ones(self.nnz_max, dtype=torch.bool, device=device))
        self.register_buffer("grow_slots", torch.empty(0, dtype=torch.long, device=device))

        # ---- bias ----
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # ---- dynamic params ----
        if self.dynamic:
            self.deltaT = int(deltaT)
            self.Tend = int(Tend)
            self.alpha = float(alpha)
            self.itr_count = 0
        else:
            self.deltaT = 0
            self.Tend = 0
            self.alpha = 0.0
            self.itr_count = 0

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        """仅用于查看：返回当前 active 的稀疏权重（detach）。"""
        with torch.no_grad():
            active_pos = torch.nonzero(self.active_mask, as_tuple=False).squeeze(1)
            idx_act = self.indices.index_select(1, active_pos).long()
            w_act = self.weights.index_select(0, active_pos)
            W = _coo(idx_act, w_act, (self.out_features, self.in_features), ref_tensor=self.weights)
            return W.detach()

    @torch.no_grad()
    def _prune_and_prepare_grow(self):
        """
        计算本轮 k，剪掉 k 条最小 |w| 的 active 边，把它们变成 grow_slots（inactive，权重置 0）。
        不改变 Parameter 对象，只做 in-place 更新。
        """
        if self.itr_count >= self.Tend:
            self.grow_slots = self.grow_slots.new_empty(0)
            return

        f_decay = self.alpha * (1 + math.cos(self.itr_count * math.pi / self.Tend)) / 2.0
        k = int(f_decay * (1 - self.sparsity) * self.nnz_max)

        active_pos = torch.nonzero(self.active_mask, as_tuple=False).squeeze(1)
        nnz_act = int(active_pos.numel())
        if nnz_act == 0 or k <= 0:
            self.grow_slots = self.grow_slots.new_empty(0)
            return

        k = min(k, nnz_act)

        w_act_abs = self.weights.index_select(0, active_pos).abs()
        _, local = torch.topk(w_act_abs, k, largest=False, sorted=False)
        prune_slots = active_pos.index_select(0, local)

        self.active_mask[prune_slots] = False
        self.weights.data[prune_slots] = 0  # 新边初始权重=0（也可改成很小随机）

        self.grow_slots = prune_slots

    def forward(self, inputs):
        if self.training and self.dynamic:
            self.itr_count += 1

        do_update = (
            self.training
            and self.dynamic
            and self.itr_count < self.Tend
            and (self.itr_count % self.deltaT == 0)
        )

        if do_update:
            self._prune_and_prepare_grow()
        else:
            self.grow_slots = self.grow_slots.new_empty(0)

        out = _DynamicSparseMM.apply(
            inputs,
            self.weights,
            self.indices,
            self.active_mask,
            self.grow_slots,
            self.out_features,
            self.in_features,
            self.max_size,
        )

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)

        return out

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, sparsity={self.sparsity:.4f}, "
            f"small_world={self.small_world}, dynamic={self.dynamic}, nnz_max={self.nnz_max}"
        )