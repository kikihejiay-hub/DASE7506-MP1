"""Ablation: no RoPE. Use absolute position embeddings instead.

Isolates the contribution of RoPE by reverting to the baseline's
absolute position embedding while keeping everything else the same.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = int(round(8 * width / 3))
        hidden = (hidden + 7) // 8 * 8
        self.hidden = hidden
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width=320, heads=8, context=256, dropout=0.25):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.context = context
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp = SwiGLU(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # No RoPE: plain attention.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        dropout = config.get("dropout", 0.25)
        self.token = nn.Embedding(config["vocab"], width)
        self.pos = nn.Embedding(self.context, width)  # absolute position
        self.blocks = nn.ModuleList(
            [Block(width, config["heads"], self.context, dropout) for _ in range(config["depth"])]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)