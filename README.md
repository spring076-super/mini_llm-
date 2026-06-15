# mini_llm —— 从零训练一个迷你大语言模型

这是一套**可以在本地真正跑起来**的小型 GPT 训练代码，对应公众号文章
《大模型不是"调"出来的：拆解 GPT、Claude 背后真正的 5 阶段流水线》里讲到的五个阶段：

| 文章里的阶段 | 对应代码 | 一句话说明 |
|---|---|---|
| 1. 数据 | `make_sample_corpus.py` | 准备原始文本语料（这里给的是占位示例） |
| 2. 分词 | `prepare_data.py` | 训练 BPE 分词器，把文本变成 token id |
| 3. 预训练 | `train.py` | 让模型学会"预测下一个 token" |
| 4. 对齐（SFT） | `sft.py` | 教模型按"提问 -> 回答"的格式说话 |
| 4. 对齐（DPO） | `dpo.py` | 用偏好对比数据，引导模型更像"更好的那个回答" |
| 5. 推理 / 评估 | `generate.py` / `evaluate.py` | 生成文本 / 算困惑度 |

代码量不大（每个文件几十到一百多行），注释也比较详细，
**目标是让你能一步步跑通整个流程、看懂每一步在干什么**，
而不是给你一个"开箱即用的大模型"。

---

## 0. 目录结构

```
mini_llm/
├── config.py            # 所有超参数：模型结构 / 训练 / SFT / DPO
├── model.py              # GPT 模型本体（注意力 + MLP + 残差）
├── utils.py               # 学习率调度、checkpoint 读写等小工具
├── data.py                # 三套数据加载逻辑（预训练 / SFT / DPO）
├── make_sample_corpus.py  # 生成占位示例语料
├── prepare_data.py        # 训练分词器 + 生成 train.bin/val.bin
├── train.py               # 预训练脚本（支持单卡和多卡 DDP）
├── sft.py                  # 监督微调
├── dpo.py                  # 偏好优化（DPO）
├── generate.py             # 文本生成 / 采样
├── evaluate.py             # 困惑度评估
├── requirements.txt
└── data/
    ├── sample_corpus.txt   # 占位语料（中英混合，纯模板生成）
    ├── sft_sample.jsonl    # SFT 示例数据
    └── dpo_sample.jsonl    # DPO 偏好数据示例
```

---

## 1. 环境准备

```bash
# 建议新建一个虚拟环境
python3 -m venv .venv
source .venv/bin/activate   # Windows 用 .venv\Scripts\activate

pip install -r requirements.txt
# 如果系统提示需要 --break-system-packages，可以加上这个参数
```

PyTorch 请按你自己的显卡情况，去 https://pytorch.org/get-started/locally/
选对应的安装命令（CPU 版 / CUDA 12.x 版等）。

---

## 2. 五分钟跑通整套流程（用占位数据）

下面这套命令在 CPU 上几分钟内就能跑完，目的是**验证整套代码没问题**，
不是真的训练出一个能用的模型（数据量和模型都太小）。

```bash
cd mini_llm

# Stage 1：生成占位语料（真实训练时跳过这一步，换成你自己的语料）
python make_sample_corpus.py --out data/sample_corpus.txt --num_docs 600

# Stage 2：训练分词器 + 生成 train.bin / val.bin
python prepare_data.py \
    --input data/sample_corpus.txt \
    --out_dir data/processed \
    --vocab_size 512

# Stage 3：预训练（这里用一个很小的模型跑 60 步）
python train.py \
    --data_dir data/processed --out_dir checkpoints/pretrain \
    --block_size 32 --n_layer 2 --n_head 4 --n_embd 64 \
    --batch_size 8 --max_iters 60 --eval_interval 20

# 看看预训练模型说话（此时还很不靠谱，纯属正常）
python generate.py \
    --ckpt checkpoints/pretrain/ckpt.pt \
    --tokenizer data/processed/tokenizer.json \
    --prompt "在山顶上，" --max_new_tokens 40 --temperature 0.8 --top_k 50

# 在整份验证集上算一下困惑度
python evaluate.py --ckpt checkpoints/pretrain/ckpt.pt --data_dir data/processed

# Stage 4a：SFT，教模型"提问 -> 回答"的格式
python sft.py \
    --init_from checkpoints/pretrain/ckpt.pt \
    --tokenizer data/processed/tokenizer.json \
    --data_path data/sft_sample.jsonl \
    --out_dir checkpoints/sft \
    --max_seq_len 32 --batch_size 4 --max_epochs 3

# Stage 4b：DPO，用偏好数据做进一步对齐
python dpo.py \
    --init_from checkpoints/sft/ckpt.pt \
    --tokenizer data/processed/tokenizer.json \
    --data_path data/dpo_sample.jsonl \
    --out_dir checkpoints/dpo \
    --max_seq_len 32 --batch_size 2 --max_epochs 1

# 用最终模型生成
python generate.py \
    --ckpt checkpoints/dpo/ckpt.pt \
    --tokenizer data/processed/tokenizer.json \
    --prompt "1加1等于多少？" --max_new_tokens 30 --temperature 0.7 --top_k 50
```

跑完这一遍，你应该能看到：
- `train.py` 的 loss / perplexity 在缓慢下降；
- `sft.py` 在 12 条样本上跑 3 个 epoch；
- `dpo.py` 打印的 `chosen>rejected 比例` 在反复波动（数据量太小，正常）；
- `generate.py` 能正常吐出文本（小模型 + 小数据，内容大概率是乱码，**这是预期的**，
  代码本身没问题）。

---

## 3. 换成真实数据训练

`data/sample_corpus.txt` 只是一份模板生成的占位语料，**不能用来训练出有意义的模型**。
真正训练时，把它换成你自己的语料即可，格式要求很简单：

- 一个 UTF-8 编码的 `.txt` 文件
- 不同"文档"之间用**一个空行**分隔（`prepare_data.py` 按空行切分文档，
  并在文档之间插入 `<eos>`）

常见的语料来源：
- 中文：维基百科中文导出、新闻语料、你自己积累的文档/笔记
- 英文：WikiText、FineWeb 等公开数据集的子集

拿到语料后，重新跑一遍 `prepare_data.py` 即可：

```bash
python prepare_data.py \
    --input data/my_corpus.txt \
    --out_dir data/processed \
    --vocab_size 8000 \
    --val_ratio 0.05
```

`vocab_size` 建议：
- 纯中文语料：8000~16000 通常够用
- 中英混合：可以适当调大到 16000~32000

---

## 4. 放大模型规模

`config.py` 里的 `GPTConfig` 默认是一个很小的配置（约几百万参数级别），
方便在 CPU 上调试。真正训练时可以按需放大，例如一个"入门级"配置：

```bash
python train.py \
    --data_dir data/processed --out_dir checkpoints/pretrain \
    --block_size 1024 \
    --n_layer 12 --n_head 12 --n_embd 768 \
    --batch_size 32 --grad_accum_steps 4 \
    --max_iters 20000 --warmup_iters 500 --lr_decay_iters 20000 \
    --learning_rate 3e-4
```

几个经验法则：
- `n_embd` 必须能被 `n_head` 整除（每个 head 的维度 = n_embd / n_head）；
- 模型变大后，单步显存占用主要由 `batch_size * block_size` 决定，
  显存不够时优先调小 `batch_size`，用 `grad_accum_steps` 补回等效 batch size；
- 关于"模型该练多少数据才够"，可以参考 Chinchilla scaling law 的经验比例：
  **训练 token 数 ≈ 20 × 参数量**。比如一个 1 亿参数的模型，
  大致需要 20 亿 token 左右的训练数据才能训练得比较充分
  （这只是一个粗略参考，不是硬性标准）。

---

## 5. 多卡训练（DDP）

`train.py` 内置了对 `torchrun` 的支持，多卡时直接用：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --data_dir data/processed --out_dir checkpoints/pretrain \
    --block_size 1024 --n_layer 12 --n_head 12 --n_embd 768 \
    --batch_size 16 --grad_accum_steps 2 \
    --max_iters 20000
```

说明：
- `--nproc_per_node` 改成你机器上的 GPU 数量；
- 此时的"等效 batch size" = `batch_size * grad_accum_steps * 卡数`；
- 这里用的是标准的 **数据并行（DDP）**：每张卡上都有一份完整模型，
  各自算各自的 batch，再同步梯度。如果模型大到单卡放不下，
  需要张量并行 / 流水线并行（Megatron-LM、DeepSpeed 等框架的能力），
  这套 mini 代码暂时没有覆盖这部分。

---

## 6. SFT / DPO 数据格式

### SFT（`data/sft_sample.jsonl`）

每行一个 JSON：

```json
{"prompt": "用一句话介绍一下你自己。", "response": "我是一个用来学习训练流程的迷你语言模型示例。"}
```

代码会把它拼成：

```
<bos>{prompt}
### 回答：
{response}<eos>
```

`prompt` 部分不计算 loss，只在 `response` 部分计算 —— 也就是只教模型
"在这种格式下该怎么回答"，而不是教它新知识（新知识主要靠 Stage 3 预训练）。

### DPO（`data/dpo_sample.jsonl`）

每行一个 JSON：

```json
{"prompt": "1加1等于多少？", "chosen": "1加1等于2。", "rejected": "这个问题好难，我不知道。"}
```

`chosen` 是"更好的回答"，`rejected` 是"更差的回答"。
`dpo.py` 会同时构造这两条序列，分别在 policy 模型和它的冻结快照
（reference 模型）上计算对数概率，损失函数会拉大两者之间的差距。

---

## 7. 各脚本参数速查

所有脚本都用 `argparse`，可以直接 `python xxx.py --help` 看到完整参数列表。
这里列一些比较常用的：

- `prepare_data.py`：`--input`、`--out_dir`、`--vocab_size`、`--val_ratio`
- `train.py`：模型结构（`--n_layer/--n_head/--n_embd/--block_size`）、
  训练步数（`--max_iters/--warmup_iters/--lr_decay_iters`）、
  `--batch_size/--grad_accum_steps`、`--resume`（断点续训）、`--compile`
- `sft.py` / `dpo.py`：`--init_from`（起始 checkpoint）、`--data_path`、
  `--max_seq_len`、`--max_epochs`、`--batch_size`
- `generate.py`：`--ckpt`、`--tokenizer`、`--prompt`、
  `--temperature`、`--top_k`、`--top_p`、`--num_samples`
- `evaluate.py`：`--ckpt`、`--data_dir`、`--split`

---

## 8. 一些实现上的小细节（给想看代码的你）

- **权重共享**：`model.py` 里 `tok_emb` 和 `lm_head` 共用一份参数矩阵
  （weight tying），是大多数 GPT 类实现的标准做法。
- **Flash Attention**：`CausalSelfAttention` 优先用
  `F.scaled_dot_product_attention`（PyTorch 2.0+ 自带），
  没有的话会退化到手写的"矩阵乘 + mask + softmax"实现。
- **SFT/DPO 的 label 对齐**：`data.py` 里的 `build_example()` 构造的
  `(input_ids, labels)` 和预训练 `get_batch()` 里的 `(x, y)` 是同一套对齐方式
  —— `labels[i]` 永远是"模型看完 `input_ids[0..i]` 之后应该预测的下一个 token"，
  prompt 部分的 `labels` 设为 `-1`（`F.cross_entropy` 的 `ignore_index`）。
- **padding 不需要额外的 attention mask**：训练时统一用右侧 padding，
  在因果注意力下，真实 token 永远不会"看到"它后面的 padding token，
  所以 padding 对 loss 没有任何污染，不需要再传一个 attention mask 给模型。
- **checkpoint 格式**：保存时把 `GPTConfig` 转成普通 `dict`（而不是直接存
  dataclass 对象），是为了兼容 PyTorch 2.6+ 默认开启的
  `torch.load(weights_only=True)`，避免加载时报"unsupported global"的错误。

---

## 9. 这套代码的边界

这是一套**教学/实验用途**的最小可运行实现，刻意省略了不少"生产级"细节，比如：

- 没有做数据去重、质量过滤（文章里提到的 FineWeb 那一套数据清洗流程）；
- 没有实现张量并行 / 流水线并行 / ZeRO 等大模型训练优化；
- DPO 部分没有实现更复杂的变体（如 IPO、KTO、GRPO 等）；
- 评估只给了困惑度，没有接入 MMLU / GPQA 这类评测基准。

如果你看完代码、跑通流程之后想往下深入，这些就是很好的"下一步"方向。
