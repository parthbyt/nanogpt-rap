import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from rapgpt.config import ModelConfig


class LearnedEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding(pos)
        return self.dropout(tok_emb + pos_emb)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)),
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.resid_dropout(y)


class FeedForward(nn.Module):
    def __init__(self, n_embed, expansion=4, dropout=0.0):
        super().__init__()
        hidden = expansion * n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed=n_embed, n_head=n_head, block_size=block_size, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed=n_embed, expansion=ffn_mult, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        vocab_size: int = 16000,
        block_size: int = 256,
        n_embed: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if config is not None:
            vocab_size = config.vocab_size
            block_size = config.block_size
            n_embed = config.n_embed
            n_head = config.n_head
            n_layer = config.n_layer
            ffn_mult = config.ffn_mult
            dropout = config.dropout

        self.block_size = block_size
        self.embedding = LearnedEmbedding(
            vocab_size=vocab_size,
            n_embed=n_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=n_embed,
                n_head=n_head,
                block_size=block_size,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def load_pretrained(cls, checkpoint_path, device="cpu", strict=True):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls()
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()

        metadata = {
            "step": checkpoint.get("step", 0),
            "val_loss": checkpoint.get("val_loss", None),
            "path": str(checkpoint_path),
        }
        return model, metadata
