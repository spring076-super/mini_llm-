"""
prepare_data.py
----------------
对应文章 Stage 1（数据）+ Stage 2（分词）。

做两件事：
  1. 用 BPE 在你的语料上训练一个 tokenizer，存成 tokenizer.json
  2. 把语料切成 train / val 两份，编码成 token id，
     存成 .bin 文件（uint16 的 numpy memmap），供 train.py 高效随机读取

用法示例：
  python prepare_data.py \
      --input data/sample_corpus.txt \
      --out_dir data/processed \
      --vocab_size 8000 \
      --val_ratio 0.1
"""

import argparse
import os

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def build_tokenizer(input_path: str, vocab_size: int, out_dir: str) -> Tokenizer:
    """用 BPE 在 input_path 上训练一个 tokenizer 并保存。"""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel pre-tokenizer：对中文也友好，不会因为没见过的字符变成 <unk>
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([input_path], trainer=trainer)

    os.makedirs(out_dir, exist_ok=True)
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[tokenizer] 词表大小: {tokenizer.get_vocab_size()}  已保存到 {tokenizer_path}")
    return tokenizer


def encode_corpus(tokenizer: Tokenizer, input_path: str, out_dir: str, val_ratio: float):
    """把语料编码成 token id，按 val_ratio 切出验证集，分别存成 .bin 文件。"""
    eos_id = tokenizer.token_to_id("<eos>")

    with open(input_path, "r", encoding="utf-8") as f:
        # 按空行分段，每一段算一篇"文档"，文档之间插入 <eos>
        text = f.read()
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    if len(docs) < 2:
        docs = [text]

    split_idx = max(1, int(len(docs) * (1 - val_ratio)))
    train_docs, val_docs = docs[:split_idx], docs[split_idx:]
    if not val_docs:  # 语料太小，至少留一篇做验证
        val_docs = train_docs[-1:]

    for split_name, split_docs in [("train", train_docs), ("val", val_docs)]:
        ids = []
        for doc in split_docs:
            ids.extend(tokenizer.encode(doc).ids)
            ids.append(eos_id)
        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(out_dir, f"{split_name}.bin")
        arr.tofile(out_path)
        print(f"[data] {split_name}: {len(arr):,} tokens -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始文本语料文件（utf-8）")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = build_tokenizer(args.input, args.vocab_size, args.out_dir)
    encode_corpus(tokenizer, args.input, args.out_dir, args.val_ratio)

    print("\n[done] 数据已就绪。real vocab_size =", tokenizer.get_vocab_size())
    print("       训练时请把 GPTConfig.vocab_size 设置为这个数字。")


if __name__ == "__main__":
    main()
