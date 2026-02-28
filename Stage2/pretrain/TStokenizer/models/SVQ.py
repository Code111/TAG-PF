from typing import Callable, Optional
import torch
from torch import nn

import torch.nn.functional as F
from ..layers.VQ import VectorQuantizer
from ..layers.encoder import Encoder
from ..layers.decoder import Decoder
from ..layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        
        # load parameters
        c_in = configs['enc_in']
        self.enc_in = configs['enc_in']
        context_window = configs['seq_len']
        e_layers = configs['e_layers']
        d_layers = configs['d_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        patch_len = configs['patch_len']
        self.patch_len = configs['patch_len']
        attn_dropout = configs['attn_dropout']
        codebook_size = configs['codebook_size']
        sparsity = configs['sparsity']
        # 检查可整除
        if context_window % patch_len != 0:
            print("上下文窗口不能整除补丁长度")
        else:
            patch_num = context_window // patch_len
            print(f'可以放入的补丁数量：{patch_num}')

        self.encoder = Encoder(patch_num, patch_len, patch_num, d_model, n_heads, d_k=None, d_v=None, d_ff=d_ff,
                        norm='BatchNorm', attn_dropout=attn_dropout, dropout=dropout, activation='gelu',
                        res_attention=True, e_layers=e_layers, pre_norm=False, store_attn=False)

        self.vq = VectorQuantizer(n_e=codebook_size, e_dim=d_model, beta=0.25, l2_norm=True, show_usage=True)

        self.decoder = Decoder(c_in, patch_num, patch_len, d_model, n_heads,sparsity, d_k=None, d_v=None, d_ff=d_ff,
                        norm='BatchNorm', attn_dropout=attn_dropout, dropout=dropout, activation='gelu',
                        res_attention=True, d_layers=d_layers, pre_norm=False, store_attn=False)
        
        self.revin_layer = RevIN(c_in, affine=False, subtract_last=False)


 
    @torch.no_grad()
    def get_token_ids(self, x: torch.Tensor) -> torch.LongTensor:
        """
        x: [B, L, C]  (C = enc_in)
        return token_ids: [B, C, n_patches]
        """
        self.eval()

        # RevIN normalize
        x, stats = self.revin_layer(x, "norm", return_stats=True)                 # [B, L, C]

        # patchify -> [B, C, n_patches, patch_len]
        x_patch = patchify(x, patch_len=self.patch_len)
        B, C, n_patches, p_len = x_patch.shape

        # encoder -> [B*C, n_patches, d_model]
        z = self.encoder(x_patch)

        # vq expects [B*C, d_model, n_patches]
        z_in = z.permute(0, 2, 1).contiguous()
        # VectorQuantizer.forward returns encoding_indices as the last item
        quant, vq_loss, commit_loss, entropy_loss, codebook_utilization, codebook_perplexity, _, encoding_indices = self.vq(z_in)

        # encoding_indices: [B*C*n_patches]  -> [B, C, n_patches]
        token_ids = encoding_indices.view(B, C, n_patches).long()
        return token_ids, stats

    @torch.no_grad()
    def ids_to_series(self, token_ids: torch.LongTensor,  stats: dict) -> torch.Tensor:
        """
        token_ids: [B, C, n_patches]  (必须与 patch_len 对应，且 codebook_size 范围内)
        return recon: [B, L, C]
        """
        self.eval()

        if token_ids.dim() != 3:
            raise ValueError(f"token_ids must be [B, C, n_patches], got {token_ids.shape}")

        B, C, n_patches = token_ids.shape
        p_len = self.patch_len

        # [B, C, n_patches] -> [B*C, n_patches]
        flat_ids = token_ids.reshape(B * C, n_patches)

        # codebook lookup -> [B*C, n_patches, d_model]
        # VectorQuantizer.get_codebook_entry supports direct indexing
        z_q = self.vq.get_codebook_entry(flat_ids)  # [B*C, n_patches, d_model]

        # decoder expects [B*C, d_model, n_patches]
        # dec_in = z_q.permute(0, 2, 1).contiguous()
        # out_patch = self.decoder(z_q)            # [B*C, n_patches, patch_len]
        out_patch = self.decoder(z_q)            # [B*C, n_patches, patch_len]

        # reshape back -> [B, C, n_patches, patch_len]
        out_patch = out_patch.view(B, C, n_patches, p_len)

        # merge patches -> [B, C, L]
        recon = out_patch.reshape(B, C, n_patches * p_len).permute(0,2,1)

        # RevIN denorm
        recon = self.revin_layer(recon, mode="denorm", stats=stats)
        return recon
    


    def forward(self, x):
        # x = self.revin_layer(x, 'norm')
        x, stats = self.revin_layer(x, "norm", return_stats=True)


        # B, L, C = x.shape
        x = patchify(x, patch_len=self.patch_len)
        B, C, n_patches, p_len = x.shape
        z = self.encoder(x)                                                      # z: [bs * nvars x patch_num x d_model]
        quant, vq_loss, commit_loss, entropy_loss, codebook_utilization, codebook_perplexity, min_encodings, group_encoding_indices_ls = self.vq(z.permute(0,2,1).contiguous())
        commit_loss1 = vq_loss + commit_loss
        z = self.decoder(quant.permute(0,2,1).contiguous())
        # 重塑回原维度 [B, C, patch_num, patch_len]
        z = z.view(B, C, n_patches, p_len)
        # 合并 patch => [B, C, L]
        recon = z.reshape(B, C, n_patches * p_len).permute(0,2,1)
        # recon = self.revin_layer(recon, 'denorm')
        # stats_1 = {"mean": stats["mean"][..., -2:-1], "stdev": stats["stdev"][..., -2:-1]}
        recon = self.revin_layer(recon, mode="denorm", stats=stats)

        return  recon, commit_loss1, codebook_utilization, codebook_perplexity



def patchify(x, patch_len):
    """
    将输入张量 x 转换为 patch 形式。

    参数：
        x: Tensor，形状通常为 (B, L, C)
        patch_len: int，每个 patch 的长度
    返回：
        Tensor，形状为 (B, C, num_patches, patch_len)
    """
    # 维度重排
    x = x.permute(0, 2, 1)
    # 展开成 patch
    x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
    return x