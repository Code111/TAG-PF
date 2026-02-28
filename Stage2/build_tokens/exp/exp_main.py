# exp/exp_main.py
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import warnings

from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from TStokenizer.models.SVQ import Model as SVQModel
from utils.deepseek_prompt_en import DeepSeekPromptEN

warnings.filterwarnings("ignore")


def _unwrap_dates(batch_dates):
    # DataLoader collate 可能把 list[str] 变成 tuple(list[str]) 或者套一层
    if len(batch_dates) == 1 and isinstance(batch_dates[0], (list, tuple)):
        dates = batch_dates[0]
    else:
        dates = batch_dates
    return [str(d) for d in dates]


def build_window_payload_from_raw_table(x_date_list, x_raw_np):
    if not isinstance(x_raw_np, np.ndarray):
        raise TypeError(f"x_raw_np must be np.ndarray, got {type(x_raw_np)}")
    if x_raw_np.ndim != 2:
        raise ValueError(f"x_raw_np must be 2D [L,4], got shape {x_raw_np.shape}")

    L, C = x_raw_np.shape
    if C != 4:
        raise ValueError(f"x_raw_np must have 4 columns [TSI,DNI,GHI,OT], got {C}")
    if len(x_date_list) != L:
        raise ValueError(f"len(x_date_list)={len(x_date_list)} but L={L}")

    rows = {
        "date":  {"start": str(x_date_list[0]),"end": str(x_date_list[-1])},
        "Total solar irradiance (W/m2)": x_raw_np[:, 0].astype(float).round(2).tolist(),
        "Direct normal irradiance (W/m2)": x_raw_np[:, 1].astype(float).round(2).tolist(),
        "Global horizontal irradiance (W/m2)": x_raw_np[:, 2].astype(float).round(2).tolist(),
        "OT (MW)": x_raw_np[:, 3].astype(float).round(2).tolist(),
    }
    return {"rows": rows}


def build_fallback_prompt_text(x_date_list):
    start = x_date_list[0] if len(x_date_list) else ""
    end = x_date_list[-1] if len(x_date_list) else ""
    return (
        "PV power forecasting task."
        f"Window: {start} - {end}."
        "Night/Day: The diurnal structure is characterized by a night regime with near-zero irradiance and OT, "
        "and a daytime regime with active generation and rise–peak–fall behavior."
        "Pattern: The weather-driven and diurnal behavior is marked by a gradual increase in irradiance and OT during "
        "the morning, followed by a peak in the afternoon and a decline in the evening. The pattern is relatively stable, "
        "with some fluctuations and persistence."
        "Goal: Use historical irradiance and OT to forecast OT for the next 24 hours."
    )


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_vq_model(self):
        with open(self.args.tokens_config, "r", encoding="utf-8") as f:
            svq_cfg = json.load(f)
        vq_model = SVQModel(svq_cfg)
        vq_model.load_state_dict(torch.load(self.args.tokens_checkpoints))
        vq_model = vq_model.to(self.device)
        vq_model.eval()
        for p in vq_model.parameters():
            p.requires_grad = False
        return vq_model

    def _get_prompt_client(self):
        if getattr(self, "_prompt_client", None) is None:
            self._prompt_client = DeepSeekPromptEN(
                api_key="EMPTY",
                base_url=self.args.llm_base_url,
                model=self.args.llm_model,
                timeout=self.args.llm_timeout,
                temperature=self.args.llm_temperature,
                max_tokens=self.args.llm_max_tokens,
                use_response_format=self.args.llm_use_response_format,
                max_retries=self.args.llm_max_retries,
                retry_sleep=self.args.llm_retry_sleep,
            )
        return self._prompt_client

    @torch.no_grad()
    def _encode_tokens(self, x, y):
        x_ids, x_stats = self.vq_model.get_token_ids(x)
        y_ids, y_stats = self.vq_model.get_token_ids(y)
        x_ids_raw = x_ids.reshape(x_ids.shape[0], -1).long()
        y_ids_raw = y_ids.reshape(y_ids.shape[0], -1).long()
        return x_ids_raw, y_ids_raw, x_stats, y_stats

    def build_tokens(self, setting):
        dataset, loader = data_provider(self.args)

        all_x_ids, all_y_ids = [], []
        all_x_scaled, all_y_scaled = [], []
        all_x_stats = {"mean": [], "stdev": []}
        all_y_stats = {"mean": [], "stdev": []}

        all_prompt_text = []
        all_prompt_is_fallback = []  # ✅ 记录哪些样本用的是 fallback（很重要）

        prompt_client = self._get_prompt_client()
        last_prompt_text = None
        for idx, batch in enumerate(tqdm(loader, desc="Building tokens"), start=0):
            batch_x_scaled, batch_y_scaled, batch_x_date, batch_y_date, batch_x_raw, batch_y_raw = batch

            batch_x_scaled = batch_x_scaled.float().to(self.device)
            batch_y_scaled = batch_y_scaled.float().to(self.device)

            x_raw_np = batch_x_raw[0].numpy()
            x_date_list = _unwrap_dates(batch_x_date)

            summary = build_window_payload_from_raw_table(x_date_list, x_raw_np)

            # ===== prompt：不达标就重试；超上限才 fallback =====

            need_prompt = (self.args.prompt_every == 0) or (idx % self.args.prompt_every == 0)
            

            if need_prompt:
                try:
                    # t0 = time.time()
                    prompt_text = prompt_client.generate_prompt_text(summary)
                    last_prompt_text = prompt_text
                    # t1 = time.time()
                    # print(f"Prompt time: {t1 - t0:.4f}s")
                    # print(prompt_text)
                    all_prompt_text.append(prompt_text)
                    all_prompt_is_fallback.append(False)
                except Exception as e:
                    print(f"[Prompt FAIL after retries] idx={idx} | fallback. Reason: {e}")
                    fallback = build_fallback_prompt_text(x_date_list)
                    # print(fallback)
                    all_prompt_text.append(fallback)
                    all_prompt_is_fallback.append(True)
            else:
                all_prompt_text.append(last_prompt_text)
                all_prompt_is_fallback.append(False)

            # ===== tokens 仍然正常生成 =====
            x_ids_raw, y_ids_raw, x_stats, y_stats = self._encode_tokens(batch_x_scaled, batch_y_scaled)

            all_x_ids.append(x_ids_raw.cpu())
            all_y_ids.append(y_ids_raw.cpu())
            all_x_scaled.append(batch_x_scaled.cpu())
            all_y_scaled.append(batch_y_scaled.cpu())

            all_x_stats["mean"].append(x_stats["mean"].detach().cpu())
            all_x_stats["stdev"].append(x_stats["stdev"].detach().cpu())
            all_y_stats["mean"].append(y_stats["mean"].detach().cpu())
            all_y_stats["stdev"].append(y_stats["stdev"].detach().cpu())

        x_ids_raw = torch.cat(all_x_ids, dim=0)
        y_ids_raw = torch.cat(all_y_ids, dim=0)
        x_scaled = torch.cat(all_x_scaled, dim=0)
        y_scaled = torch.cat(all_y_scaled, dim=0)

        x_stats = {k: torch.cat(v, dim=0) for k, v in all_x_stats.items()}
        y_stats = {k: torch.cat(v, dim=0) for k, v in all_y_stats.items()}

        save_path = os.path.join(self.args.root_path, self.args.save_data_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        meta = {
            "data_path": self.args.data_path,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,
            "patch_len": self.args.patch_len,
            "enc_in": self.args.enc_in,
            "target_col": self.args.target_col,
            "Tin": int(x_ids_raw.shape[1]),
            "Tout": int(y_ids_raw.shape[1]),
            "codebook_size": int(getattr(self.vq_model.vq, "n_e", -1)),
            "llm_max_retries": self.args.llm_max_retries,
            "llm_use_response_format": bool(self.args.llm_use_response_format),
        }

        torch.save(
            {
                "meta": meta,
                "x_ids_raw": x_ids_raw,
                "y_ids_raw": y_ids_raw,
                "x_scaled": x_scaled,
                "y_scaled": y_scaled,
                "x_stats": x_stats,
                "y_stats": y_stats,
                "prompt_text": all_prompt_text,
                "prompt_is_fallback": torch.tensor(all_prompt_is_fallback, dtype=torch.bool),
            },
            save_path,
        )

        print(f"[Saved] {save_path}")
        print(f"meta={meta}")
        print(f"[Prompt] fallback_count={sum(all_prompt_is_fallback)} / {len(all_prompt_is_fallback)}")
