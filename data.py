"""
data.py
-------
三套数据加载逻辑，分别对应 Stage 3（预训练）与 Stage 4（对齐：SFT / DPO）。

1. get_batch()       —— 预训练用。直接在 .bin 文件（token id 的 memmap）上
                         随机截取 (x, y) 片段，x 是输入，y 是 x 整体右移一位
                         （也就是"下一个 token 是什么"）。

2. SFTDataset        —— 读取 jsonl: {"prompt": ..., "response": ...}
                         拼接成  <bos> prompt 模板 + response <eos>
                         prompt 部分的 label 设为 -1，loss 只在 response 上计算。

3. DPODataset        —— 读取 jsonl: {"prompt": ..., "chosen": ..., "rejected": ...}
                         分别构造 chosen / rejected 两条序列，复用上面同一套
                         build_example() 逻辑。

两个 collate_fn 负责把一个 batch 里长度不同的序列 pad 到同一长度。
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


SFT_TEMPLATE = "{prompt}\n### 回答：\n"


def load_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


# ---------------------------------------------------------------------------
# 1. 预训练数据：memmap 随机采样
# ---------------------------------------------------------------------------

def get_batch(data_dir, split, block_size, batch_size, device):
    """
    从 train.bin / val.bin 里随机取 batch_size 个长度为 block_size 的片段。
    返回 x, y，y 是 x 整体右移一位（即每个位置的"下一个 token"）。
    """
    path = os.path.join(data_dir, f"{split}.bin")
    # 用 memmap 而不是一次性 load 到内存，数据量大时也不会爆内存
    data = np.memmap(path, dtype=np.uint16, mode="r")

    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"{path} 里只有 {len(data)} 个 token，不够切出 block_size={block_size} 的片段，"
            f"请减小 block_size 或增加语料。"
        )

    ix = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data[i: i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1: i + 1 + block_size].astype(np.int64) for i in ix])

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# 公共：把 (prompt, completion) 编码成 (input_ids, labels)
# ---------------------------------------------------------------------------

def build_example(tokenizer: Tokenizer, prompt: str, completion: str, max_seq_len: int):
    """
    构造 (input_ids, labels)，长度都是 len(full_ids) - 1，且严格满足：
        labels[i] 是 "模型看到 input_ids[0..i] 之后，应该预测的下一个 token"
    这和预训练 get_batch() 里 x / y 的对齐方式完全一致——
    forward() 内部不做任何移位，全靠调用方把 (input, target) 对齐好。

    prompt 部分对应的 label 设为 -1（不参与 loss），
    只有 completion（response / chosen / rejected）部分的 label 是真实 token id。
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    prompt_text = SFT_TEMPLATE.format(prompt=prompt)
    prompt_ids = tokenizer.encode(prompt_text).ids
    completion_ids = tokenizer.encode(completion).ids + [eos_id]

    # input_ids/labels 的长度 = len(prompt_ids) + len(completion_ids)
    # （= len(full_ids) - 1，其中 full_ids = [bos] + prompt_ids + completion_ids）
    budget = max_seq_len
    total = len(prompt_ids) + len(completion_ids)
    if total > budget:
        overflow = total - budget
        if overflow < len(prompt_ids):
            # 优先从 prompt 头部裁掉，尽量保留完整的 completion
            prompt_ids = prompt_ids[overflow:]
        else:
            # prompt 全部裁掉后还不够，再从 completion 头部裁
            extra = overflow - len(prompt_ids)
            prompt_ids = []
            completion_ids = completion_ids[extra:] or [eos_id]

    full_ids = [bos_id] + prompt_ids + completion_ids
    prompt_len = 1 + len(prompt_ids)  # 含 <bos> 在内，prompt 部分的 token 数

    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    for i in range(prompt_len - 1):
        labels[i] = -1

    return input_ids, labels


def _pad_batch(seqs, pad_value, max_len=None):
    max_len = max_len or max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# 2. SFT 数据集
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """jsonl 每行: {"prompt": "...", "response": "..."}"""

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, labels = build_example(
            self.tokenizer, ex["prompt"], ex["response"], self.max_seq_len
        )
        return input_ids, labels


def sft_collate_fn(batch, pad_id: int):
    input_ids, labels = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    x = _pad_batch(input_ids, pad_value=pad_id, max_len=max_len)
    y = _pad_batch(labels, pad_value=-1, max_len=max_len)
    attn_mask = (x != pad_id).long()
    return x, y, attn_mask


# ---------------------------------------------------------------------------
# 3. DPO 数据集
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """jsonl 每行: {"prompt": "...", "chosen": "...", "rejected": "..."}"""

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        chosen_ids, chosen_labels = build_example(
            self.tokenizer, ex["prompt"], ex["chosen"], self.max_seq_len
        )
        rejected_ids, rejected_labels = build_example(
            self.tokenizer, ex["prompt"], ex["rejected"], self.max_seq_len
        )
        return chosen_ids, chosen_labels, rejected_ids, rejected_labels


def dpo_collate_fn(batch, pad_id: int):
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = zip(*batch)
    max_len = max(
        max(len(x) for x in chosen_ids),
        max(len(x) for x in rejected_ids),
    )
    out = {
        "chosen_ids": _pad_batch(chosen_ids, pad_id, max_len),
        "chosen_labels": _pad_batch(chosen_labels, -1, max_len),
        "rejected_ids": _pad_batch(rejected_ids, pad_id, max_len),
        "rejected_labels": _pad_batch(rejected_labels, -1, max_len),
    }
    out["chosen_mask"] = (out["chosen_ids"] != pad_id).long()
    out["rejected_mask"] = (out["rejected_ids"] != pad_id).long()
    return out
