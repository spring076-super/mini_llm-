"""
dpo.py
------
DPO（直接偏好优化），对应文章 Stage 4 第二步里提到的"更轻量的对齐方法"。

核心思路：
  - policy 模型：从 SFT checkpoint 出发，是要被训练的模型
  - reference 模型：policy 在训练开始那一刻的"快照"，全程冻结，
    作用是防止 policy 为了讨好偏好数据而"跑偏"太远
  - 对每条样本算出 chosen / rejected 两个回答在 policy 和 reference
    下的对数概率，损失函数鼓励：
        (policy更偏好chosen的程度) > (reference更偏好chosen的程度)

  loss = -log sigmoid( beta * [
            (logP_policy(chosen) - logP_policy(rejected))
          - (logP_ref(chosen)    - logP_ref(rejected))
         ] )

用法：
  python dpo.py \
      --init_from checkpoints/sft/ckpt.pt \
      --tokenizer data/processed/tokenizer.json \
      --data_path data/dpo_sample.jsonl \
      --out_dir checkpoints/dpo
"""

import argparse
import copy
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DPOConfig, GPTConfig
from data import DPODataset, dpo_collate_fn, load_tokenizer
from model import GPT
from utils import resolve_device, resolve_dtype, save_checkpoint


def sequence_logprobs(model, input_ids, labels):
    """
    返回每个样本里 "label != -1" 那些位置的 log-prob 之和，shape: (B,)
    即整段 completion（chosen 或 rejected）在当前模型下的对数概率。
    """
    logits, _ = model(input_ids)  # (B, T, V)，不传 targets，所以不会算 loss
    log_probs = F.log_softmax(logits, dim=-1)

    mask = (labels != -1)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0  # 占位，避免 gather 时索引为 -1 报错

    token_logp = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask  # 屏蔽 prompt / padding 部分
    return token_logp.sum(dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default=DPOConfig.init_from, help="SFT checkpoint 路径")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data_path", default=DPOConfig.data_path)
    p.add_argument("--out_dir", default=DPOConfig.out_dir)

    p.add_argument("--beta", type=float, default=DPOConfig.beta)
    p.add_argument("--learning_rate", type=float, default=DPOConfig.learning_rate)
    p.add_argument("--weight_decay", type=float, default=DPOConfig.weight_decay)
    p.add_argument("--grad_clip", type=float, default=DPOConfig.grad_clip)
    p.add_argument("--max_epochs", type=int, default=DPOConfig.max_epochs)
    p.add_argument("--batch_size", type=int, default=DPOConfig.batch_size)
    p.add_argument("--max_seq_len", type=int, default=DPOConfig.max_seq_len)
    p.add_argument("--log_interval", type=int, default=DPOConfig.log_interval)
    p.add_argument("--device", default=DPOConfig.device)
    p.add_argument("--dtype", default=DPOConfig.dtype)
    p.add_argument("--seed", type=int, default=DPOConfig.seed)
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

    # ---- policy 模型：从 SFT checkpoint 加载，可训练 ----
    ckpt = torch.load(args.init_from, map_location=device)
    model_config = GPTConfig(**ckpt["model_config"])
    if args.max_seq_len > model_config.block_size:
        raise ValueError(
            f"--max_seq_len ({args.max_seq_len}) 超过了模型的 block_size "
            f"({model_config.block_size})。"
        )
    policy = GPT(model_config).to(device)
    policy.load_state_dict(ckpt["model"])
    print(f"[info] policy 模型加载完成，参数量 {policy.num_params():,}")

    # ---- reference 模型：policy 的冻结快照 ----
    reference = copy.deepcopy(policy).to(device)
    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()

    optimizer = policy.configure_optimizer(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )

    dataset = DPODataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(dpo_collate_fn, pad_id=pad_id),
    )
    print(f"[info] DPO 偏好样本数: {len(dataset)}")

    policy.train()
    step = 0
    for epoch in range(args.max_epochs):
        for batch in loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            with autocast_ctx:
                policy_chosen_logp = sequence_logprobs(policy, chosen_ids, chosen_labels)
                policy_rejected_logp = sequence_logprobs(policy, rejected_ids, rejected_labels)

                with torch.no_grad():
                    ref_chosen_logp = sequence_logprobs(reference, chosen_ids, chosen_labels)
                    ref_rejected_logp = sequence_logprobs(reference, rejected_ids, rejected_labels)

                pi_logratios = policy_chosen_logp - policy_rejected_logp
                ref_logratios = ref_chosen_logp - ref_rejected_logp
                logits = pi_logratios - ref_logratios

                loss = -F.logsigmoid(args.beta * logits).mean()

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                with torch.no_grad():
                    # "偏好准确率"：policy 给 chosen 打的分是不是确实比 rejected 高
                    acc = (pi_logratios > 0).float().mean().item()
                print(
                    f"epoch {epoch} | step {step:5d} | loss {loss.item():.4f} "
                    f"| chosen>rejected 比例 {acc:.2f}"
                )
            step += 1

    save_checkpoint(args.out_dir, policy, optimizer, model_config, iter_num=step)
    print(f"[done] DPO checkpoint 已保存到 {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
