import argparse
import os
import random
import numpy as np


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_bool_arg(p: argparse.ArgumentParser, name: str, default: bool, help_text: str = ""):
    """
    Create paired flags: --name / --no-name
    """
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(f"--{name}", dest=name, action="store_true", help=help_text + f" (default={default})")
    g.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {name}")
    p.set_defaults(**{name: default})


def get_args():
    p = argparse.ArgumentParser("LLM for TS Tokens (FSDP, bf16-only)")

    # ---------------- basic ----------------
    p.add_argument("--random_seed", type=int, default=2026)
    p.add_argument("--model", type=str, default="qwen4ts")
    p.add_argument("--checkpoints", type=str, default="./checkpoints")
    p.add_argument("--result_file", type=str, default="result.txt")

    # ---------------- data ----------------
    p.add_argument(
        "--root_path",
        type=str,
        default="../build_tokens/dataset-pre/wind_farms",
    )
    p.add_argument("--data_path", type=str, default="128-Solar station site 1 (Nominal capacity-50MW).pt")

    # ---------------- llm ----------------
    p.add_argument("--LLM_model_path", type=str, default="LLM/Qwen3-0.6B-Base")
    p.add_argument("--codebook_size", type=int, default=128)

    # p.add_argument("--mean_resizing", type=bool, default=False, help="HF resize_token_embeddings(mean_resizing=...) if supported")
    add_bool_arg(p, "mean_resizing", True, "HF resize_token_embeddings(mean_resizing=...) if supported")
    add_bool_arg(p, "print_trainable", True, "Print trainable parameters on rank0")

    # 是否训练新增 vocab 行（建议多卡 FULL_SHARD 关闭）
    add_bool_arg(p, "train_new_vocab_rows", True, "Train ONLY the newly added vocab rows (via grad mask hooks)")

    # ---------------- SVQ decode model ----------------
    p.add_argument(
        "--tokens_config",
        type=str,
        default="./config/128-config.json",
    )
    p.add_argument(
        "--tokens_checkpoints",
        type=str,
        default="../build_tokens/checkpoints/Solar station site 1 (Nominal capacity-50MW).csv/128-checkpoint.pth",
    )

    # ---------------- train ----------------
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.000035)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    add_bool_arg(p, "gradient_checkpointing", False, "Enable gradient checkpointing (memory saver)")

    # ---------------- split ----------------
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # ---------------- generate eval ----------------
    add_bool_arg(p, "do_generate_eval", True, "Run generate-eval AFTER training finishes (recommended)")
    add_bool_arg(p, "gen_eval_each_epoch", False, "Run generate-eval EVERY epoch (slow)")

    p.add_argument("--gen_max_new_tokens", type=int, default=24)  # kept
    add_bool_arg(p, "gen_use_cache", True, "Use KV cache for generation")

    # ---------------- precision ----------------
    add_bool_arg(p, "bf16", True, "bf16-only: must be True")
    add_bool_arg(p, "fp16", False, "Compatibility flag only (NOT supported)")

    # ---------------- dataloader runtime ----------------
    p.add_argument("--num_workers", type=int, default=0)
    add_bool_arg(p, "pin_memory", True, "DataLoader pin_memory")
    add_bool_arg(p, "persistent_workers", False, "DataLoader persistent_workers")
    p.add_argument("--prefetch_factor", type=int, default=2)

    # collator
    p.add_argument("--pad_to_multiple_of", type=int, default=8)
    add_bool_arg(p, "enable_prompt_cache", True, "Enable prompt tokenize LRU cache")
    p.add_argument("--prompt_cache_max_items", type=int, default=2048)

    # ---------------- LoRA flags ----------------
    add_bool_arg(p, "lora_enable", True, "Enable LoRA (requires peft)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    # ---------------- FSDP wrap policy ----------------
    p.add_argument("--fsdp_min_num_params", type=int, default=10_000_000)
    add_bool_arg(p, "fsdp_auto_wrap", False, "Enable FSDP auto-wrap (multi-GPU)")
    add_bool_arg(p, "sync_module_states", False, "FSDP sync_module_states")

    # dist timeout (minutes)
    p.add_argument("--dist_timeout_min", type=int, default=60)

    return p.parse_args()


def _require_torchrun_env(args) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))

    if rank < 0 or local_rank < 0 or world_size <= 0:
        raise RuntimeError(
            "本项目为 torchrun 启动（单卡也建议 torchrun）：\n"
            "单卡：CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 run_longExp.py [args...]\n"
            "双卡：CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run_longExp.py [args...]"
        )

    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.distributed = True
    args.gpu = local_rank


def _init_dist(args):
    """
    Initialize NCCL process group once (ONLY here), with explicit device_id to avoid hangs.
    """
    import datetime
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP-only 版本需要 CUDA/NCCL；当前未检测到可用 GPU。")

    torch.cuda.set_device(args.local_rank)

    timeout = datetime.timedelta(minutes=int(getattr(args, "dist_timeout_min", 60)))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timeout,
        device_id=torch.device(f"cuda:{args.local_rank}"),
    )
    dist.barrier(device_ids=[args.local_rank])


def _enforce_precision_args(args):
    if bool(getattr(args, "fp16", False)):
        raise RuntimeError("本版本为 bf16-only：请不要传 --fp16。")
    if not bool(getattr(args, "bf16", True)):
        raise RuntimeError("本版本为 bf16-only：请不要传 --no-bf16。")


def _set_stable_sdpa_backend():
    """
    更稳定的 SDPA 路径：禁用 flash/mem_efficient，强制 math。
    目的：减少 bf16 下非确定性（尤其 greedy generate 对微小差异很敏感）。
    """
    import torch

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        # older torch 可能不支持这些开关
        pass

    # 一般也建议关 tf32（更一致）
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def main():
    args = get_args()
    _require_torchrun_env(args)

    import torch
    import torch.distributed as dist
    from exp.exp_main import Exp_Main

    _enforce_precision_args(args)

    # 更稳定的 attention backend（入口统一设置）
    # _set_stable_sdpa_backend()

    _init_dist(args)

    args.use_gpu = True
    seed_everything(int(args.random_seed) + int(args.rank))

    if args.rank == 0:
        print("\n--- Configuration (FSDP, bf16-only) ---")
        print(args)
        print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    print(f"[rank{args.rank}] local_rank={args.local_rank} using cuda:{args.local_rank}", flush=True)

    try:
        exp = Exp_Main(args)
        setting = (
            f"llm_{args.data_path.split('.')[0]}"
            f"_bs{args.train_batch_size}"
            f"_lr{args.learning_rate}"
            f"_cb{args.codebook_size}"
            f"_wd{args.weight_decay}"
            f"_warm{args.warmup_ratio}"
            f"_ls{args.label_smoothing}"
            f"_gc{args.grad_clip}"
            f"_Lmp{args.LLM_model_path}"
        )
        if args.rank == 0:
            print(f"\n>>>> Starting training: {setting} <<<<")
        exp.train(setting)
        torch.cuda.empty_cache()

    finally:
        try:
            dist.barrier(device_ids=[args.local_rank])
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    # os.environ["WORLD_SIZE"] = "1"          # 总进程数
    # os.environ["RANK"] = "0"                # 全局进程排名
    # os.environ["LOCAL_RANK"] = "0"          # 本地进程排名
    # os.environ["MASTER_ADDR"] = "127.0.0.1" # 主节点地址
    # os.environ["MASTER_PORT"] = "29500"     # 主节点端口
    main()
