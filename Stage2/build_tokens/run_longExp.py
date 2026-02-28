# run_longExp.py
import argparse
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build TS tokens (.pt)")

    parser.add_argument("--random_seed", type=int, default=2026)

    # tokenizer (SVQ) config + ckpt
    parser.add_argument(
        "--tokens_config",
        type=str,
        default="./build_tokens/config/config.json",
    )
    parser.add_argument(
        "--tokens_checkpoints",
        type=str,
        default="./checkpoints/Solar station site 8 (Nominal capacity-30MW).csv/128-checkpoint.pth",
    )

    # data
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset-pre/solar_stations",
    )
    parser.add_argument("--data_path", type=str, default="Solar station site 8 (Nominal capacity-30MW).csv")
    parser.add_argument("--save_data_path", type=str, default="Solar station site 8 (Nominal capacity-35MW).pt")

    # lengths
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)

    # for tokenizer
    parser.add_argument("--patch_len", type=int, default=4)     # 必须和 tokenizer 训练一致
    parser.add_argument("--enc_in", type=int, default=4)        # 输入变量数（建议与 tokenizer 训练一致）
    parser.add_argument("--target_col", type=int, default=-1)   # 目标列索引：默认最后一列

    # dataloader
    parser.add_argument("--num_workers", type=int, default=0)

    # GPU（注意：不用 type=bool，避免 argparse 坑）
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    # prompt (本地 vLLM)
    parser.add_argument("--llm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_model", type=str, default="Qwen3-235B-AWQ")
    parser.add_argument("--llm_temperature", type=float, default=0.9)
    parser.add_argument("--llm_max_tokens", type=int, default=500)
    parser.add_argument("--llm_timeout", type=int, default=10)
    parser.add_argument("--llm_use_response_format", action="store_true", default=False)

    # Retry policy
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--llm_retry_sleep", type=float, default=0.1)


    parser.add_argument("--prompt_every", type=int, default=96)

    args = parser.parse_args()
    set_seed(args.random_seed)

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = [int(x) for x in args.devices.split(",")]
        args.device_ids = device_ids
        args.gpu = device_ids[0]

    print("Args:")
    print(args)

    setting = f"build_tokens_sl{args.seq_len}_pl{args.patch_len}_pred{args.pred_len}"
    exp = Exp_Main(args)
    exp.build_tokens(setting)
    torch.cuda.empty_cache()
