"""
config.py
----------
所有可调超参数集中在这里。包括：
  - GPTConfig：模型结构相关
  - TrainConfig：预训练相关
  - SFTConfig：有监督微调相关
  - DPOConfig：偏好对齐相关

设计原则：
  1. 默认值跑在 CPU / 单张消费级显卡上也能跑通（用于学习和调试）。
  2. 真正训练时，把 n_layer / n_embd / block_size 等调大即可，
     代码逻辑完全不用改。
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # ---- 词表 & 序列长度 ----
    vocab_size: int = 8000      # 由 tokenizer 决定，训练完 tokenizer 后会回填
    block_size: int = 256       # 上下文窗口长度（最大序列长度）

    # ---- 模型规模 ----
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    ff_dim: int = 4 * 384       # FFN 隐层维度，一般是 n_embd 的 4 倍

    # ---- 正则化 ----
    dropout: float = 0.1
    bias: bool = False          # Linear/LayerNorm 是否带 bias，False 更省参数也更稳定


@dataclass
class TrainConfig:
    # ---- 数据 ----
    data_dir: str = "data/processed"   # 存放 train.bin / val.bin 的目录

    # ---- 优化器 ----
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- 训练步数 & 学习率调度 ----
    max_iters: int = 3000
    warmup_iters: int = 200
    lr_decay_iters: int = 3000

    # ---- batch ----
    batch_size: int = 32        # 单卡 micro-batch
    grad_accum_steps: int = 1   # 梯度累积步数，等效 batch = batch_size * grad_accum_steps * world_size

    # ---- 评估 & 日志 ----
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20

    # ---- 其他 ----
    out_dir: str = "checkpoints/pretrain"
    seed: int = 1337
    device: str = "auto"        # "auto" | "cpu" | "cuda" | "cuda:0" ...
    dtype: str = "auto"         # "auto" | "float32" | "bfloat16" | "float16"
    compile_model: bool = False  # torch.compile，CPU 上建议关闭


@dataclass
class SFTConfig:
    data_path: str = "data/sft_sample.jsonl"
    init_from: str = "checkpoints/pretrain/ckpt.pt"
    out_dir: str = "checkpoints/sft"

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    max_epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 256

    log_interval: int = 5
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337


@dataclass
class DPOConfig:
    data_path: str = "data/dpo_sample.jsonl"
    init_from: str = "checkpoints/sft/ckpt.pt"
    out_dir: str = "checkpoints/dpo"

    beta: float = 0.1            # DPO 温度系数，越小越"激进"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    max_epochs: int = 1
    batch_size: int = 2
    max_seq_len: int = 256

    log_interval: int = 5
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337
