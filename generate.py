"""
generate.py
-----------
加载一个 checkpoint，对应文章"中场：怎么让它开口说话"——
用 temperature / top_k / top_p 控制采样的"创造力"。

用法：
  python generate.py \
      --ckpt checkpoints/pretrain/ckpt.pt \
      --tokenizer data/processed/tokenizer.json \
      --prompt "在山顶上，" \
      --max_new_tokens 60 --temperature 0.8 --top_k 50
"""

import argparse

import torch

from config import GPTConfig
from data import load_tokenizer
from model import GPT
from utils import resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    prompt_ids = [bos_id] + tokenizer.encode(args.prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"[prompt] {args.prompt!r}")
    print(f"[config] temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}")
    print("-" * 60)

    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"[sample {i + 1}]\n{text}\n")


if __name__ == "__main__":
    main()
