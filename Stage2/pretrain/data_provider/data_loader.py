# data_provider/data_loader.py
import os
import torch
from torch.utils.data import Dataset


class TokensPTDataset(Dataset):
    def __init__(self, root_path: str, data_path: str):
        self.path = os.path.join(root_path, data_path)
        data = torch.load(self.path, map_location="cpu")
        

        self.input_tokens = data["x_ids_raw"]
        self.output_tokens = data["y_ids_raw"]
        self.batch_y = data["y_scaled"]
        self.out_stats = data["y_stats"]

        self.prompt_text = data.get("prompt_text", None)
        self.prompt_is_fallback = data.get("prompt_is_fallback", None)

        self.N = int(self.input_tokens.shape[0])

        self.meta = data.get("meta", {}) or {}
        self.meta["Tin"] = int(self.input_tokens.reshape(self.N, -1).shape[-1])

        out = self.output_tokens
        if out.dim() in (2, 3):
            self.meta["Tout"] = int(out.shape[-1])
        else:
            raise ValueError(f"Unexpected output_tokens dim: {tuple(out.shape)}")

        self.meta["pred_len"] = int(self.batch_y.shape[1])
   

    def __len__(self):
        return self.N

    def _safe_take_stat(self, stat_value, idx: int) -> torch.Tensor:
        if torch.is_tensor(stat_value):
            v = stat_value
            if v.dim() >= 1 and v.size(0) == self.N:
                v = v[idx]
            return v.detach().cpu()
        else:
            return torch.as_tensor(stat_value).detach().cpu()

    def __getitem__(self, idx: int):
        x = self.input_tokens[idx].reshape(-1).long()
        y = self.output_tokens[idx].reshape(-1).long()
        by = self.batch_y[idx].float()

        y_stats = {}
        if isinstance(self.out_stats, dict):
            for k, stat_value in self.out_stats.items():
                y_stats[k] = self._safe_take_stat(stat_value, idx)

        item = {
            "x_ids_raw": x,
            "y_ids_raw": y,
            "batch_y": by,
            "y_stats": y_stats,
        }

        if self.prompt_text is not None:
            pt = self.prompt_text[idx]
            if isinstance(pt, bytes):
                pt = pt.decode("utf-8", errors="ignore")
            item["prompt_text"] = str(pt)

        if self.prompt_is_fallback is not None:
            pf = self.prompt_is_fallback[idx]
            if torch.is_tensor(pf):
                item["prompt_is_fallback"] = bool(pf.item())
            else:
                item["prompt_is_fallback"] = bool(pf)

        return item
