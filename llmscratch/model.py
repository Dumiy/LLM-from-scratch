"""The settled Llama-style architecture (Chapter 4), as a reusable module.

This is the *given* model for every chapter after Ch.4 — the one you load
`modern.pt` into. Parameter names match exactly what Ch.4/5 saved, so
checkpoints load with `strict=True`.

Teaching note: Ch.4 builds RoPE / RMSNorm / GQA / SwiGLU as `TODO`s — that's
where you *learn* them. This module is the finished version for *reuse*; import
it in downstream chapters instead of re-pasting the classes.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

GPT2_VOCAB_SIZE = 50257


@dataclass
class Config:
    vocab_size: int = GPT2_VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1408
    n_layers: int = 8
    block_size: int = 256
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide into heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"


# ---- RoPE (rotary position embeddings) ----
def precompute_rope(d_k, max_pos, theta=10000.0, device="cpu"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
    t = torch.arange(max_pos, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.d_k = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_k, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.d_k, cfg.d_model, bias=False)
        self.o_proj.RESIDUAL_SCALE_INIT = True
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, is_causal=True):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), present


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.down.RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, is_causal=True):
        a, present = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache, is_causal)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class LlamaGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.n_layers = cfg.n_layers
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        d_k = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(d_k, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE_INIT", False):
                std /= math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x, _ = block(x, cos, sin, is_causal=True)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.7, top_k=50, eot_id=None):
    """KV-cached autoregressive generation. idx: (1, P) -> (1, P+new). Stops at eot_id if given."""
    model.eval()
    caches = [None] * len(model.blocks)

    def step(tokens, start, is_causal):
        T = tokens.shape[1]
        cos, sin = model.rope_cos[start:start + T], model.rope_sin[start:start + T]
        x = model.wte(tokens)
        for i, b in enumerate(model.blocks):
            x, caches[i] = b(x, cos, sin, past_kv=caches[i], use_cache=True, is_causal=is_causal)
        return model.lm_head(model.norm_f(x))[:, -1, :]

    pos = idx.shape[1]
    logits = step(idx, 0, True)
    for _ in range(max_new_tokens):
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eot_id is not None and nxt.item() == eot_id:
            break
        logits = step(nxt, pos, False)
        pos += 1
    model.train()
    return idx
