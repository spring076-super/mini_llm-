"""
train.py
--------
预训练脚本，对应文章 Stage 3（训练：预测下一个 token）。

单卡 / CPU：
  python train.py --data_dir data/processed --max_iters 1000

单机多卡（DDP，对应文章里提到的"多 GPU"场景）：
  torchrun --standalone --nproc_per_node=4 train.py --data_dir data/processed

核心训练循环就是文章里那张"4 步循环图"：
  ① 取一批 token  -> ② 模型预测下一个 token 的分布
  -> ③ 和真实 token 比较算 loss -> ④ 反向传播更新参数
  -> 重复
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config import GPTConfig, TrainConfig
from data import get_batch, load_tokenizer
from model import GPT
from utils import get_lr, is_ddp, resolve_device, resolve_dtype, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()

    # 数据 / 输出
    p.add_argument("--data_dir", type=str, default=TrainConfig.data_dir)
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)

    # 模型结构（不传就用 config.py 里的默认值）
    p.add_argument("--vocab_size", type=int, default=None, help="不传则自动从 tokenizer.json 读取")
    p.add_argument("--block_size", type=int, default=GPTConfig.block_size)
    p.add_argument("--n_layer", type=int, default=GPTConfig.n_layer)
    p.add_argument("--n_head", type=int, default=GPTConfig.n_head)
    p.add_argument("--n_embd", type=int, default=GPTConfig.n_embd)
    p.add_argument("--dropout", type=float, default=GPTConfig.dropout)

    # 优化 / 训练步数
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)
    p.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--min_lr", type=float, default=TrainConfig.min_lr)
    p.add_argument("--max_iters", type=int, default=TrainConfig.max_iters)
    p.add_argument("--warmup_iters", type=int, default=TrainConfig.warmup_iters)
    p.add_argument("--lr_decay_iters", type=int, default=TrainConfig.lr_decay_iters)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)

    # 评估 / 日志
    p.add_argument("--eval_interval", type=int, default=TrainConfig.eval_interval)
    p.add_argument("--eval_iters", type=int, default=TrainConfig.eval_iters)
    p.add_argument("--log_interval", type=int, default=TrainConfig.log_interval)

    # 其他
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--dtype", type=str, default=TrainConfig.dtype)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--compile", action="store_true", default=TrainConfig.compile_model)
    p.add_argument("--resume", action="store_true", help="从 out_dir/ckpt.pt 继续训练")

    return p.parse_args()


def main():
    args = parse_args()

    # ----------------------- DDP（多卡）初始化 -----------------------
    ddp = is_ddp()
    if ddp:
        init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
        is_master = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_world_size = 1
        is_master = True
        seed_offset = 0
        device = resolve_device(args.device)

    torch.manual_seed(args.seed + seed_offset)

    dtype = resolve_dtype(args.dtype, device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # ----------------------- 词表大小 -----------------------
    vocab_size = args.vocab_size
    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    if vocab_size is None and os.path.exists(tokenizer_path):
        vocab_size = load_tokenizer(tokenizer_path).get_vocab_size()
        if is_master:
            print(f"[info] 从 {tokenizer_path} 读取到 vocab_size = {vocab_size}")
    if vocab_size is None:
        vocab_size = GPTConfig.vocab_size

    # ----------------------- 构建模型 -----------------------
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        ff_dim=4 * args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_config).to(device)
    if is_master:
        print(f"[info] 模型参数量: {model.num_params():,}")

    optimizer = model.configure_optimizer(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(TrainConfig.beta1, TrainConfig.beta2),
    )

    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter_num", 0)
        if is_master:
            print(f"[info] 从 checkpoint 恢复，起始 iter = {start_iter}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if torch.cuda.is_available() else None)

    use_fp16_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=use_fp16_scaler)

    # ----------------------- 评估函数 -----------------------
    @torch.no_grad()
    def estimate_loss():
        raw_model = model.module if ddp else model
        raw_model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(args.data_dir, split, args.block_size, args.batch_size, device)
                with autocast_ctx:
                    _, loss = raw_model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        raw_model.train()
        return out

    # ----------------------- 训练循环 -----------------------
    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = get_lr(
            it,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 梯度累积：把多个 micro-batch 的 loss 平均后再 backward
        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch(args.data_dir, "train", args.block_size, args.batch_size, device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)

            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if is_master and it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"iter {it:6d} | loss {loss.item() * args.grad_accum_steps:.4f} "
                f"| lr {lr:.2e} | {dt * 1000 / max(1, args.log_interval):.1f} ms/iter"
            )

        # ---- 周期性评估 + 保存 ----
        if is_master and (it % args.eval_interval == 0 or it == args.max_iters - 1) and it > start_iter:
            losses = estimate_loss()
            ppl_train = torch.exp(torch.tensor(losses["train"])).item()
            ppl_val = torch.exp(torch.tensor(losses["val"])).item()
            print(
                f"  [eval] iter {it} | train loss {losses['train']:.4f} (ppl {ppl_train:.1f}) "
                f"| val loss {losses['val']:.4f} (ppl {ppl_val:.1f})"
            )
            save_checkpoint(args.out_dir, model, optimizer, model_config, it)

    if is_master:
        save_checkpoint(args.out_dir, model, optimizer, model_config, args.max_iters)
        print(f"[done] checkpoint 已保存到 {args.out_dir}/ckpt.pt")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
