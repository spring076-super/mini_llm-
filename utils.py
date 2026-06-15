"""
utils.py
--------
训练脚本之间共享的小工具：
  - 设备 / 精度自动选择
  - 学习率调度（warmup + cosine decay）
  - checkpoint 保存与加载
"""

import math
import os
from dataclasses import asdict, is_dataclass

import torch


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def get_lr(it, *, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """
    标准的 warmup + cosine decay 学习率调度：
      1) 0 ~ warmup_iters：线性从 0 升到 learning_rate
      2) warmup_iters ~ lr_decay_iters：余弦衰减到 min_lr
      3) 之后保持 min_lr 不变
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(out_dir, model, optimizer, model_config, iter_num, extra=None, name="ckpt.pt"):
    os.makedirs(out_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model  # 兼容 DDP
    cfg = asdict(model_config) if is_dataclass(model_config) else model_config
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_config": cfg,  # 存成普通 dict，避免 torch.load(weights_only=True) 的限制
        "iter_num": iter_num,
    }
    if extra:
        ckpt.update(extra)
    path = os.path.join(out_dir, name)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def is_ddp() -> bool:
    """是否在 `torchrun` 启动的多进程环境下运行。"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ
