import torch
import torch.nn.functional as F
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer(f"codebook_used", torch.zeros(65536))
        self.prob_alpha = 0.01
        self.register_buffer("embed_prob", torch.zeros(self.n_e))

    def forward(self, z):
        z = torch.einsum('b c n -> b n c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = -torch.sum(z_flattened ** 2, dim=1, keepdim=True) - \
            torch.sum(embedding ** 2, dim=1) + 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        # encoding
        _, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        z_q = embedding[encoding_indices].view(z.shape)


        perplexity = None
        min_encodings = None
        vq_loss = 0
        commit_loss = 0
        entropy_loss = 0
        codebook_usage = 0

        if self.show_usage:
            if self.training:
                cur_len = encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
                # compute loss for embedding
            else:
                codebook_usage = len(torch.unique(encoding_indices)) / self.n_e
                vq_loss = 0
        else:
            codebook_usage = 0


        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            # entropy_loss = 0

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        quant = torch.einsum('b n c -> b c n', z_q)

        # # resample by used frequency EMA
        min_encodings = torch.nn.functional.one_hot(encoding_indices, num_classes=self.n_e).to(dtype=z.dtype, device=z.device)
        avg_probs = torch.mean(min_encodings, dim=0)  # prob of used codes
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # e^-H(p) entropy of distribution p
        if self.training:
            updated_prob = torch.lerp(self.embed_prob, avg_probs, self.prob_alpha)
            self.embed_prob.copy_(updated_prob)
            norm_distance = F.softmax((d - d.max()).t(), dim=1)
            # check NaN / Inf
            if not torch.isfinite(norm_distance).all():
                print("Warning: probs contain NaN or Inf, replacing with uniform distribution")
                norm_distance = torch.ones_like(norm_distance) / norm_distance.numel()
            # avoid zero probability
            norm_distance = torch.clamp(norm_distance, min=1e-10)
            norm_distance /= torch.sum(norm_distance)
            # resample by distance
            prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
            random_feat = z_flattened.detach()[prob]
            freq_prob = 1 - torch.exp((-(updated_prob * self.n_e * 10) / self.prob_alpha) - 0.001).unsqueeze(1).repeat(1, self.e_dim)
            self.embedding.weight.data \
                = self.embedding.weight.data * freq_prob + random_feat * (1 - freq_prob)

        return quant, vq_loss, commit_loss, entropy_loss, codebook_usage, perplexity, min_encodings, encoding_indices

    def get_codebook_entry(self, indices, shape=None, channel_first=True, groups_to_use=0):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        quant = embedding[indices]
        return quant


