# exp/exp_basic.py
import os
import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model, self.tokenizer = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _is_rank0(self) -> bool:
        return int(getattr(self.args, "rank", 0)) == 0

    def _acquire_device(self):
        use_gpu = bool(getattr(self.args, "use_gpu", False)) and torch.cuda.is_available()

        if use_gpu:
            env_local_rank = os.environ.get("LOCAL_RANK", None)
            gpu_idx = int(env_local_rank) if env_local_rank is not None else int(getattr(self.args, "gpu", 0))

            torch.cuda.set_device(gpu_idx)
            device = torch.device(f"cuda:{gpu_idx}")

            if bool(getattr(self.args, "bf16", True)) and (not torch.cuda.is_bf16_supported()):
                raise RuntimeError("bf16-only 运行需要当前 GPU 支持 bf16，但检测到不支持。")

            if self._is_rank0():
                print(f"Use GPU: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
            return device

        if self._is_rank0():
            print("Use CPU")
        return torch.device("cpu")
