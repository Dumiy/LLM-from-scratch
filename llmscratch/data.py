"""Reusable data plumbing: token-cache builder, the memmap batch loader, the
background prefetcher, and the weighted-mixture loader.

These are the settled versions from Chapters 3a / 4 — import them instead of
re-pasting `get_batch` into every notebook.
"""
import os
import time
import threading
import queue

import numpy as np
import torch


def build_token_cache(path, repo, token_budget, enc, eot, config=None, split="train", text_field="text"):
    """Stream `repo`, tokenize `text_field`, write a flat uint16 .bin of ~token_budget tokens.
    Skips entirely if `path` already exists (tokenizing is expensive -- do it once)."""
    if os.path.exists(path):
        n = os.path.getsize(path) // 2
        print(f"{path} already exists ({n:,} tokens) -- skipping rebuild")
        return path
    from datasets import load_dataset
    ds = load_dataset(repo, name=config, split=split, streaming=True)
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(token_budget,))
    written, t0 = 0, time.time()
    for doc in ds:
        text = doc.get(text_field) or ""
        if not text:
            continue
        ids = enc.encode_ordinary(text) + [eot]
        take = ids[: token_budget - written]
        arr[written:written + len(take)] = take
        written += len(take)
        if written >= token_budget:
            break
    arr.flush()
    print(f"done: {written:,} tokens -> {path} ({time.time()-t0:.0f}s)")
    return path


def get_batch(data, block_size, batch_size, device="cpu"):
    """data: 1-D uint16 array (memmap). Returns (x, y) int64 tensors on `device`.
    y is x shifted +1 (next-token targets). Uses pinned async H2D on CUDA."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    if str(device).startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


class BatchPrefetcher:
    """Background thread that keeps `get_batch` results ready so the GPU never waits on the CPU.
    One thread suffices (numpy/pin/copy release the GIL); avoids multiprocessing-DataLoader pain
    on Windows. Producer and consumer share the default CUDA stream, so the enqueued copy is
    ordered before the kernels that consume it."""
    def __init__(self, data, block_size, batch_size, device="cpu", depth=4):
        self.data, self.block_size, self.batch_size, self.device = data, block_size, batch_size, device
        self.q = queue.Queue(maxsize=depth)
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        while not self._stop:
            try:
                self.q.put(get_batch(self.data, self.block_size, self.batch_size, self.device), timeout=0.5)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()

    def close(self):
        self._stop = True


def mixture_get_batch(mix, weights, block_size, batch_size, device="cpu"):
    """Pick a source by `weights`, then a random (x, y) window from that source's memmap.
    `mix` is a list of dicts each with a "data" memmap (Chapter 4's data mixture)."""
    src = mix[np.random.choice(len(mix), p=weights)]
    return get_batch(src["data"], block_size, batch_size, device)
