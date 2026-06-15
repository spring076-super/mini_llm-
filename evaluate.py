"""
evaluate.py
-----------
对应文章 Stage 5（评估）里"预训练阶段看困惑度"的部分：

  困惑度 = exp(平均 loss)
  数值越低，说明模型对这份数据的"惊讶程度"越低，也就是预测得越准。

和 train.py 训练过程中"抽样估计"不同，这里会把整个 split（比如 val.bin）
切成不重叠的 block_size 片段，过一遍模型，给出更准确的整体困惑度。

用法：
  python evaluate.py --ckpt checkpoints/pretrain/ckpt.pt \
      --data_dir data/processed --split val
"""

import argparse

import numpy as np
import torch

from config import GPTConfig
from model import GPT
from utils import resolve_device, resolve_dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = GPTConfig(**ckpt["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    block_size = model_config.block_size
    data = np.memmap(f"{args.data_dir}/{args.split}.bin", dtype=np.uint16, mode="r")

    n_chunks = len(data) // (block_size + 1)
    if n_chunks == 0:
        raise ValueError("数据太短，不够切出一个 block_size 的片段。")

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_chunks, args.batch_size):
            ids = range(start, min(start + args.batch_size, n_chunks))
            xs, ys = [], []
            for i in ids:
                offset = i * (block_size + 1)
                chunk = data[offset: offset + block_size + 1].astype(np.int64)
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
            x = torch.tensor(np.stack(xs), device=device)
            y = torch.tensor(np.stack(ys), device=device)

            _, loss = model(x, y)
            n_tok = x.numel()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

    avg_loss = total_loss / total_tokens
    ppl = float(np.exp(avg_loss))
    print(f"[{args.split}] tokens={total_tokens:,}  avg_loss={avg_loss:.4f}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
