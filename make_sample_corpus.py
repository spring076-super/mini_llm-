"""
make_sample_corpus.py
----------------------
生成一份很小的"占位"中英双语语料，仅用于：
  - 跑通 tokenizer 训练
  - 跑通预训练流程，验证 loss / perplexity 是否在下降

⚠️ 这不是真实的训练数据！
真正训练时，请把 data/sample_corpus.txt 换成你自己的语料，例如：
  - 中文：维基百科中文导出、新闻语料、你自己的文档
  - 英文：WikiText、FineWeb 的子集等

用法：
  python make_sample_corpus.py --out data/sample_corpus.txt --num_docs 600
"""

import argparse
import random

ZH_SUBJECTS = ["小狐狸", "程序员", "老猫", "学生", "厨师", "旅行者", "工程师", "孩子", "诗人", "机器人"]
ZH_ACTIONS = ["跳过了篱笆", "写了一段代码", "煮了一锅汤", "读完了一本书",
              "修好了电脑", "画了一幅画", "解决了一个难题", "弹了一首曲子",
              "整理了房间", "训练了一个模型"]
ZH_PLACES = ["在公园里", "在深夜的办公室", "在小镇的咖啡馆", "在山顶上",
              "在安静的图书馆", "在拥挤的地铁上", "在阳台上", "在实验室里"]
ZH_CONNECTORS = ["然后", "接着", "随后", "不久之后"]

EN_SUBJECTS = ["the quick fox", "a curious student", "an old robot", "the new engineer",
               "a tired traveler", "the young chef", "a clever cat", "the quiet poet"]
EN_ACTIONS = ["jumped over the fence", "wrote a few lines of code", "cooked a warm soup",
              "finished reading a book", "fixed the broken laptop", "painted a small picture",
              "solved a tricky puzzle", "trained a tiny model"]
EN_PLACES = ["in the park", "late at night in the office", "at a small cafe in town",
              "on top of the hill", "in a quiet library", "on a crowded train",
              "on the balcony", "in the lab"]
EN_CONNECTORS = ["Then", "After that", "Soon", "A moment later"]


def make_zh_doc(rng: random.Random) -> str:
    sentences = []
    for _ in range(rng.randint(2, 4)):
        subj = rng.choice(ZH_SUBJECTS)
        act = rng.choice(ZH_ACTIONS)
        place = rng.choice(ZH_PLACES)
        conn = rng.choice(ZH_CONNECTORS)
        sentences.append(f"{place}，{subj}{act}。{conn}，大家都觉得很有趣。")
    return "".join(sentences)


def make_en_doc(rng: random.Random) -> str:
    sentences = []
    for _ in range(rng.randint(2, 4)):
        subj = rng.choice(EN_SUBJECTS)
        act = rng.choice(EN_ACTIONS)
        place = rng.choice(EN_PLACES)
        conn = rng.choice(EN_CONNECTORS)
        sentences.append(f"{place.capitalize()}, {subj} {act}. {conn}, everyone thought it was great.")
    return " ".join(sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_corpus.txt")
    parser.add_argument("--num_docs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs = []
    for i in range(args.num_docs):
        if i % 2 == 0:
            docs.append(make_zh_doc(rng))
        else:
            docs.append(make_en_doc(rng))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(docs))

    print(f"已生成 {len(docs)} 篇占位文档 -> {args.out}")


if __name__ == "__main__":
    main()
