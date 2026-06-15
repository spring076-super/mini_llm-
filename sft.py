"""
sft.py
------
监督微调（Stage 4 第一步：SFT）。

从预训练 checkpoint 出发，在 (prompt, response) 数据上继续训练：
  - 输入拼成  <bos> + prompt模板 + response + <eos>
  - prompt 部分的 label 设为 -1，loss 只在 response 部分计算
  - 也就是文章里说的"教会模型说话的格式"，而不是教新知识

用法：
  python sft.py \
      --init_from checkpoints/pretrain/ckpt.pt \
      --tokenizer data/processed/tokenizer.json \
      --data_path data/sft_sample.jsonl \
      --out_dir checkpoints/sft \
      --max_epochs 3
"""

import argparse
from contextlib import nullcontext
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import GPTConfig, SFTConfig
from data import SFTDataset, load_tokenizer, sft_collate_fn
from model import GPT
from utils import resolve_device, resolve_dtype, save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default=SFTConfig.init_from, help="预训练 checkpoint 路径")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data_path", default=SFTConfig.data_path)
    p.add_argument("--out_dir", default=SFTConfig.out_dir)

    p.add_argument("--learning_rate", type=float, default=SFTConfig.learning_rate)
    p.add_argument("--weight_decay", type=float, default=SFTConfig.weight_decay)
    p.add_argument("--grad_clip", type=float, default=SFTConfig.grad_clip)
    p.add_argument("--max_epochs", type=int, default=SFTConfig.max_epochs)
    p.add_argument("--batch_size", type=int, default=SFTConfig.batch_size)
    p.add_argument("--max_seq_len", type=int, default=SFTConfig.max_seq_len)
    p.add_argument("--log_interval", type=int, default=SFTConfig.log_interval)
    p.add_argument("--device", default=SFTConfig.device)
    p.add_argument("--dtype", default=SFTConfig.dtype)
    p.add_argument("--seed", type=int, default=SFTConfig.seed)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    tokenizer = load_tokenizer(args.tokenizer)
    pad_id = tokenizer.token_to_id("<pad>")

    # ---- 加载预训练模型 ----
    ckpt = torch.load(args.init_from, map_location=device)
    model_config = GPTConfig(**ckpt["model_config"])
    if args.max_seq_len > model_config.block_size:
        raise ValueError(
            f"--max_seq_len ({args.max_seq_len}) 超过了预训练模型的 block_size "
            f"({model_config.block_size})，请调小 max_seq_len。"
        )
    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[info] 从 {args.init_from} 加载预训练权重，参数量 {model.num_params():,}")

    optimizer = model.configure_optimizer(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(sft_collate_fn, pad_id=pad_id),
    )
    print(f"[info] SFT 样本数: {len(dataset)}")

    # ---- 训练循环 ----
    model.train()
    step = 0
    for epoch in range(args.max_epochs):
        for x, y, _attn_mask in loader:
            # 说明：右侧 padding + 因果注意力下，真实 token 永远看不到它后面的
            # padding token，所以这里不需要再额外传 attention mask 给模型。
            x, y = x.to(device), y.to(device)

            with autocast_ctx:
                _, loss = model(x, y)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                print(f"epoch {epoch} | step {step:5d} | loss {loss.item():.4f}")
            step += 1

    save_checkpoint(args.out_dir, model, optimizer, model_config, iter_num=step)
    print(f"[done] SFT checkpoint 已保存到 {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
