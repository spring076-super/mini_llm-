"""
model.py
--------
一个标准的 Decoder-only Transformer（GPT 结构）。

对应文章里的 Stage 3（训练阶段使用的核心架构）：
  token embedding + position embedding
    -> N 层 [LayerNorm -> 多头自注意力(因果掩码) -> 残差
              LayerNorm -> MLP(GELU)            -> 残差]
    -> LayerNorm
    -> lm_head（线性层，输出每个 token 在词表上的 logits）

这套结构同时也是 SFT / DPO 阶段复用的对象——
对齐阶段不改架构，只改"喂给它什么数据、用什么损失函数"。
"""

import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力。"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 一次性算出 q, k, v 三份投影
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # 没有 flash attention 时的兜底因果掩码
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        # PyTorch >= 2.0 自带高效实现
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd

        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=True,  # 自动施加"只能看过去"的因果掩码
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
            y = attn @ v

        # 把多头结果拼回去
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    """两层全连接 + GELU，俗称 FeedForward。"""

    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, config.ff_dim, bias=config.bias)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(config.ff_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """一个 Transformer block：注意力 + MLP，都带残差连接。"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN 结构：先归一化再进子层，训练更稳定
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """完整模型：embedding -> N 个 Block -> 输出头。"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享（weight tying）：输入 embedding 和输出投影共用一份参数矩阵
        # 既节省参数，也是大多数 GPT 系实现的标准做法
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T)  输入 token id
        targets: (B, T)  下一个 token 的真实 id（预训练 / SFT 都用这个接口）
                  其中值为 -1 的位置不参与 loss 计算（用于 SFT 时屏蔽 prompt 部分）
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"序列长度 {T} 超过 block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                  top_k=None, top_p=None, eos_token_id=None):
        """
        自回归生成。对应文章"中场：怎么让它开口说话"那一节。

        idx:          (B, T) 起始 token（prompt）
        temperature:  采样温度，越大越随机
        top_k:        只在概率最高的 k 个 token 里采样
        top_p:        只在累积概率达到 p 之前的 token 里采样（nucleus sampling）
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 超过 block_size 就裁掉最早的部分，只保留最近的上下文
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                # 累积概率超过 top_p 之后的 token 全部丢弃
                drop_mask = cum_probs - probs > top_p
                sorted_logits = sorted_logits.masked_fill(drop_mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        把参数分两组：
          - 二维以上的权重（Linear/Embedding）参与 weight decay
          - 一维的参数（LayerNorm、bias）不参与
        这是目前社区训练 GPT 类模型的常见做法。
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # 如果当前 PyTorch 支持 fused AdamW 且在 GPU 上，就用 fused 版本（更快）
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
