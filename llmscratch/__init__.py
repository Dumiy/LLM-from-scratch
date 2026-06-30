"""llmscratch — shared, settled building blocks for the LLM-from-scratch chapters.

One canonical source of truth so notebooks stop re-pasting (and drifting from) the
same model/data code. Import the pieces a chapter *reuses*; still implement a
chapter's own `TODO`s inline (that's the lesson).

    from llmscratch.model import Config, LlamaGPT, generate
    from llmscratch.data import get_batch, build_token_cache

The model classes match the parameter names that Chapters 4-6 saved their
checkpoints with, so `modern.pt` / `moe.pt` load without surgery.
"""
from .model import (
    Config,
    RMSNorm,
    GroupedQueryAttention,
    SwiGLU,
    LlamaBlock,
    LlamaGPT,
    precompute_rope,
    rotate_half,
    apply_rope,
    repeat_kv,
    generate,
)
from .data import get_batch, build_token_cache, BatchPrefetcher, mixture_get_batch

__all__ = [
    "Config", "RMSNorm", "GroupedQueryAttention", "SwiGLU", "LlamaBlock", "LlamaGPT",
    "precompute_rope", "rotate_half", "apply_rope", "repeat_kv", "generate",
    "get_batch", "build_token_cache", "BatchPrefetcher", "mixture_get_batch",
]
