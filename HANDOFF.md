# Handoff

Continuity notes for the next Claude Code session on this project. Read this first.

## Project goal

Relearn ML/AI fundamentals from first principles, building toward an LLM from scratch. Stated end goal (README): hopefully build a state-of-the-art LLM — but see "Reality check" below, since the user also wants it multilingual and agentic (function-calling), trained locally.

## Hardware / environment

- GPU: NVIDIA RTX 4060 Ti, 8GB VRAM. Driver supports up to CUDA 13.2.
- `requirements.txt` pins `torch>=2.6` + `jupyter>=1.0` + `datasets` + `tiktoken` + `matplotlib` + `tqdm>=4.66` via `--extra-index-url https://download.pytorch.org/whl/cu132`, plus `triton-windows>=3.0 ; platform_system == "Windows"` (Triton isn't bundled with PyTorch on Windows — needed for `torch.compile`).
- Installed and verified: `torch==2.12.1+cu132`, `torch.cuda.is_available() == True`, device detected as the RTX 4060 Ti. `triton-windows==3.7.1.post27` installed and verified (`torch.compile` runs a real CUDA kernel end-to-end).
- 8GB VRAM is the binding constraint going forward — expect to need LoRA/QLoRA, gradient checkpointing, and/or small model sizes (tens to low hundreds of millions of params) for anything beyond toy runs.
- Storage: ~300GB usable on disk. FineWeb-2/CulturaX/MADLAD-400 are multi-TB datasets in full — never download them whole. Budget: ~6-8 languages × ~2B tokens each ≈ 50-80GB raw text (1 token ≈ 4 bytes). Pull only the per-language subset (e.g. `fra_Latn`, `deu_Latn`) via `streaming=True`, stop at the token budget, tokenize to binary, then delete the raw text shards.

## What's done — Chapter 1: The Transformer

`chapters/chapter-1-transformer/transformer.ipynb` — an encoder-decoder transformer built from raw PyTorch tensor ops (no `nn.Transformer`/`nn.MultiheadAttention`). Structured as 8 phases, each: concept explanation (with worked numeric examples, analogies, ASCII diagrams) → `TODO` stub → self-check assertion cell. The user implemented all 8 phases themselves; all checks pass:

1. Scaled dot-product attention
2. Multi-head attention
3. Positional encoding
4. Position-wise feed-forward
5. Encoder layer
6. Decoder layer
7. Encoder / decoder stacks
8. Causal masking + full `Transformer`

**Decision made:** this from-scratch code was for learning only. Going forward (any future chapter/training work), use PyTorch's built-ins (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`/`DecoderLayer`, or `F.scaled_dot_product_attention`) rather than the hand-rolled version — faster, correct, fused kernels, and the learning goal is already met.

## What's done — Chapter 2: Training a Decoder-Only Model (GPT-style)

`chapters/chapter-2-decoder-llm/gpt.ipynb` — a GPT-style decoder-only model, same TODO-stub + concept + check-cell pattern as Chapter 1. Per the Chapter 1 decision, the attention *math* itself uses `F.scaled_dot_product_attention(is_causal=True)` rather than being re-derived — the new ground covered here is genuinely GPT-specific:

1. Tokenizer (`tiktoken` GPT-2 BPE, reused not retrained) + data pipeline on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (streamed, ~6.6M tokens cached, well under the storage budget)
2. Causal self-attention (fused `qkv` projection, `is_causal=True`)
3. GPT block — **pre-norm** (`x + sublayer(norm(x))`), not Chapter 1's post-norm; GELU not ReLU
4. Full GPT model — learned positional embeddings (not sinusoidal), weight tying (`lm_head.weight = wte.weight`), and a from-scratch-discovered necessity: **explicit weight init**
5. Training loop (next-token-prediction cross-entropy) + the overfit-tiny-batch sanity check
6. Autoregressive generation (temperature + top-k sampling)
7. A real training run: ~17.6M params, 2000 steps, ~2.3 min on the RTX 4060 Ti, loss 10.8 → 2.7, produces recognizable (if imperfect) TinyStories-style text. Now also records `loss_history` and plots a training curve (per-step loss + 50-step moving average + dashed `ln(V)` baseline) via matplotlib — the plot cell is *given* scaffolding, not a TODO (visualization is tooling, not the learning goal).

**Explanatory depth added (per user's repeated demand for beginner-clarity):** Phases 5 and 6 now carry variable-by-variable tables (shape + meaning for `x`/`y`/`logits`/`loss`, and `idx`/`idx_cond`/`probs`/`next_id`), a line-by-line walk of `zero_grad`/`backward`/`step`/`.item()`, and **worked numeric examples** for temperature (logits `[2,1,0]` at T=0.5/1.0/2.0 → actual softmax outputs) and top-k. This user consistently wants worked examples + explicit "what each variable represents" + analogies, not just prose — keep that bar for any new chapter.

**Real bug found and fixed during this chapter, worth knowing about:** training a 6-layer pre-norm model with PyTorch's *default* init gave a first-batch loss of 157 instead of the expected ~10.8 (`ln(vocab_size)`) — pre-norm residual streams accumulate variance with depth if every sublayer write isn't scaled down. Fixed with GPT-2-style init: `N(0, 0.02)` everywhere, with the two "writes back to the residual stream" per block (`out_proj`, `mlp_proj`) additionally scaled by `1/sqrt(2*n_layers)`. This fix is baked into the notebook's `GPT._init_weights` (given code, not a TODO) — if you write a training script outside this notebook later, **do not forget this**, it's not optional at any real depth.

**VRAM note:** batch_size 64 at this model size (`d_model=256, n_layers=6, block_size=128`) measured ~7.7GB peak — too close to this card's 8GB. Dropped to batch_size 32 (~4.0GB peak) in the shipped notebook.

**Editing gotcha (bit us once):** the user's open notebook tab in the editor saved a stale copy *over* programmatic notebook edits, silently reverting them. Before editing `gpt.ipynb`, tell the user to close and reopen the tab, and re-Read the file from disk after edits to confirm they stuck. Also: `NotebookEdit` markdown cells must contain real newlines, not literal `\n` escape sequences (recurring slip).

## What's done — Chapter 3: Instruction Model (Path 1, from scratch)

**Decisions locked this session:** Path 1 (pretrain a small model from scratch, then SFT) + task = English instruction/chat. Base size ~50M (`d_model=512, n_heads=8, d_ff=2048, n_layers=8, block_size=256`). SFT data: Dolly-15k to build/iterate, smol-smoltalk to scale.

Two notebooks in `chapters/chapter-3-instruction-model/`, same concept→TODO→check pattern, full explanatory depth (variable tables, worked numeric examples, particularities). The GPT classes are reproduced *given* (verbatim from Ch.2, not TODOs — already learned). **TODOs are the new ground only.**

- **`3a-pretrain.ipynb`** — turns Ch.2's toy into a real base. New TODOs (all now IMPLEMENTED, see status below): FineWeb-Edu (`sample-10BT`) streaming → uint16 `np.memmap` token cache; `get_batch` random-window loader; AMP `train_step` (**bf16**, no GradScaler — Ada supports bf16); warmup+cosine `get_lr`; full train loop with checkpoint save/resume. Given: GPT classes, gradient-checkpointing plumbing (`model.use_checkpoint`, off by default — flip only on OOM), `generate`/`complete`, real-run + plot cells. Writes `data/base.pt`. **This session it was heavily upgraded** (instrumentation + speed) — details under "Chapter 3a session work" below. Ends with a granular end-to-end **appendix** (data path, training step, forward pass, backward, generation, optimization stack, VRAM budget) as ASCII diagrams.
- **`3b-sft.ipynb`** (26 cells) — SFT into a chat model. **Core concept = loss masking** (label prompt tokens `-100`, train only on response; `cross_entropy(ignore_index=-100)`). New TODOs: `format_example` (Alpaca/Dolly text template, no new special tokens, EOT as end-of-turn), `encode_example` (build masked labels), `collate` (dynamic right-pad), `sft_train` loop. Given: GPT classes, base-load, `chat()` generation (stops at EOT). Loads `data/base.pt` → writes `data/chat.pt`.

**Key technical decisions baked into the notebooks (don't relitigate):**
- bf16 autocast, not fp16 → no GradScaler needed.
- Alpaca-style **text** chat template, not added role tokens → no embedding-table surgery on tiny data. (`### Instruction: / ### Response:`, EOT = end-of-turn.)
- **Right**-padding + `is_causal=True` → no separate attention padding mask needed (real tokens never attend to trailing pads; pad positions are `-100` in labels). This breaks with left-padding — documented in 3b Phase 3.
- SFT uses low LR (~3e-5, ~10× below pretrain) + 1-3 epochs → avoid catastrophic forgetting.
- The Ch.2 residual-init fix (`1/sqrt(2*n_layers)` on `out_proj`/`mlp_proj`) carries over and matters MORE at n_layers=8.

**Status (updated this session):** **3a is implemented and running.** The user worked the 3a TODOs (`get_batch`, `train_step`, `get_lr`, `train`) and the notebook is producing a real base. **3b TODOs (`format_example`, `encode_example`, `collate`, `sft_train`) are still mostly stubs** — `sft_train` was rewritten with the same instrumentation but the data-pipeline TODOs above it are not yet implemented, so 3b won't run end-to-end until the user fills those in. Token caches built on disk (gitignored): `fineweb_tiny.bin` (2M), `fineweb_train.bin` (200M), `fineweb_train_500M.bin` (500M). `data/base.pt` exists (checkpointed every 1000 steps).

### Chapter 3a session work — instrumentation + speed (don't redo, build on)

The 3a training loop was upgraded well past the original stub. What's in it now:
- **Instrumentation:** `train()` returns a `history` dict (`step/loss/lr/grad_norm/batch_time/vram_gb` + `val_step/val_loss`). `train_step` returns `(loss, grad_norm)` — grad norm is the pre-clip total from `clip_grad_norm_`, logged as the health signal. tqdm bar shows live `loss/lr/gnorm/ms/tok_s/vram` + ETA. At each `eval_interval` it measures val loss, checkpoints, AND generates from `SAMPLE_PROMPTS` (samples-at-validation). 4-panel dashboard (`plot_history`): loss (train+val+lnV), grad-norm (log), VRAM, step-time — all on the real-step x-axis.
- **Speed (model was launch-bound, ~4% GPU util in the naive version → ~2× after):**
  - `torch.compile(model)` — the biggest win; auto-skips if Triton missing (`_triton_available()`). NOTE Windows needed `pip install triton-windows` (done).
  - `BatchPrefetcher` — a daemon thread runs `get_batch` (0.5 ms) off the hot path so the GPU never waits. Single thread is enough (numpy/pin/copy release the GIL); deliberately NOT a multiprocessing DataLoader (Windows spawn pain).
  - TF32 (`matmul.allow_tf32`, `cudnn.allow_tf32`) + `cudnn.benchmark` in the setup cell.
  - Measured: `get_batch 0.5 ms` vs `fwd+bwd ~147 ms` (profiling cell confirms it's purely GPU-bound now); ~11k → ~28–42k tok/s.
- **Two correctness fixes `torch.compile` forced:** `save_ckpt` unwraps `_orig_mod.` (a compiled model's state_dict prefixes keys — would break 3b's `load_state_dict`); `train()` runs eval+generation on the **uncompiled** `base = getattr(model,"_orig_mod",model)` to avoid recompiles on variable-length sampling.
- **Resume-and-extend:** run cell has `RESUME` (continue interrupted run of the SAME horizon → no LR jump) and the token-cache path encodes the budget (`fineweb_train_{N}M.bin`) so raising `TOKEN_BUDGET` rebuilds instead of silently reusing a smaller cache. Bumped `BATCH_SIZE` 24→32 (peak ~6.2 GB at 32, fits 8 GB).

**Training reality (measured):** at batch 24, 6000 steps (~37M tokens, well under 1 epoch) → val loss ~4.69, coherent-fragment generations. That's **undertrained, not broken** — Chinchilla ~20 tok/param ⇒ ~1B tokens for this 51M model. Current run targets val ~3.x via 30000 steps on the 500M cache (~3 h at batch 32, ~280 ms/step). To go further: raise `MAX_STEPS` (and `TOKEN_BUDGET` to avoid repetition). A ~4.7-loss base makes a weak chat model — push the base down before judging 3b output.

**Editing note:** this session edited via live `NotebookEdit` successfully (the old stale-open-tab gotcha didn't bite, but the tool did once report "file modified since read" — re-Read before editing if that happens). `NotebookEdit` markdown cells must contain real newlines, not literal `\n`.

## What's done — Chapter 4: Modern Architecture + broader data

`chapters/chapter-4-modern-architecture/llama.ipynb` (31 cells, authored this session). **Decided by the user after seeing 3b's output** (SFT worked — format+stop learned — but the base is knowledge-poor: "capital of France" → "Germany", plus repetition loops). User's call: don't paper over it downstream; fix the base via a better architecture + broader data. Same concept→TODO→check pattern, full explanatory depth.

Rebuilds the Ch.2/3 GPT-2-style model into a **Llama-style** decoder. The five architecture changes (each a TODO with deep "what/why/benefit" markdown):
- **RoPE** (rotary position) replaces learned absolute `wpe` — relative, length-extrapolating, zero-param. TODOs: `precompute_rope`, `rotate_half`, `apply_rope`.
- **RMSNorm** replaces LayerNorm — scale-only, no mean-centering, no bias.
- **SwiGLU** replaces GELU MLP — gated; `d_ff=1408 ≈ 8/3·d_model` keeps params equal (3 matrices vs 2).
- **GQA** (`n_kv_heads=2`, 8 query heads) replaces MHA — 4× smaller KV cache. Includes `repeat_kv`.
- **No biases** anywhere.
- **KV-cache `generate`** (given) — O(N) not O(N²) decoding; the payoff of GQA.

Plus a **broader data mixture** (§7): per-source uint16 caches (FineWeb web 0.45 / FineWeb-Edu 0.35 / the-stack-smol code 0.20) + a weighted `mixture_get_batch` (TODO). Note: this is the **multilingual on-ramp** — swap sources for language slices, same loader. §8 trains the modern model (reuses 3a's instrumented loop, given) and **A/Bs against `base.pt`** via a compact `OldGPT` loader.

**Verified this session:** all 7 checks pass with reference solutions (ran a validation script with the TODOs filled in) — RoPE norm+relative property, RMSNorm, SwiGLU param parity, GQA shapes, model builds at **48.3M params** (vs 51.1M base — comparable) with fresh loss ~10.9 ≈ ln(V), **KV-cache generation bit-identical to uncached**, mixture batches. So the chapter is internally consistent; the TODO stubs are correct-by-construction.

**Reference Config:** `d_model=512, n_heads=8, n_kv_heads=2, d_ff=1408, n_layers=8, block_size=256, rope_theta=10000`. block_size kept at 256 (same as base) for a clean A/B — RoPE *enables* longer but we don't use it, to isolate the comparison to architecture+data.

**Status:** notebook authored + internally verified, **not yet run by the user** (no `modern.pt`, no mixture caches built — the FineWeb/the-stack streaming+tokenize is the slow one-time part). TODOs are stubs for the user to implement. Built via a stdlib-json builder script (`scratchpad/build_ch4.py`) + a normalize pass to add cell ids; validated with `nbformat`.

**Editing note for ch4:** the architecture pieces (RoPE/RMSNorm/SwiGLU/GQA/mixture loader) are the TODOs; the block/model/KV-cache-generate/train-loop/A-B are *given*. Reference solutions were written + run in a throwaway scratchpad script this session to confirm the checks pass — deliberately **not** committed (would spoil the TODO learning pattern). The notebook ships stubs only.

## What's done — Chapter 5: Function-Calling / Tool Use

`chapters/chapter-5-function-calling/tools.ipynb` (25 cells, authored this session). **User chose this** (over multilingual/scale/DPO) as the agentic end-goal milestone. The framing that ties the whole project together: a ~50M model can't be a knowledge store (the recurring "France→Germany" failure), so stop trying — teach it to **call a tool** that knows, with the tools supplied in the prompt so the skill generalizes.

SFTs the **Ch.4 modern base** (`modern.pt`) on **real** data: `Salesforce/xlam-function-calling-60k` (the user explicitly chose real data over a synthetic toolset). Each xLAM row = `query` + `tools` (JSON list) + `answers` (JSON call list). New TODOs (concept→TODO→check):
- `format_example` — Alpaca-style template extended with a `### Tools:` block; response = the `answers` JSON.
- `encode_example` — 3b-style loss masking, train only on the call+EOT.
- `parse_and_execute` — extract the first balanced JSON from model output, dispatch to real Python tool fns (defensive: handles single-dict calls, unknown tools, non-JSON → error, never crashes).
- `chat_with_tools` — the agentic loop: prompt → generate call → parse → execute → grounded result.

Given: the full Ch.4 LlamaGPT stack (reproduced so it stands alone) + KV-cache `generate`, `collate`, the instrumented `sft_train`, the real tool implementations (`search` over a tiny fact dict, `calculator` via safe `ast` eval, `get_weather` stub) + `TOOL_SCHEMAS`. Loads `modern.pt` → writes `data/toolcaller.pt`.

**Key technical decisions baked in (don't relitigate):**
- **Real data = xLAM-60k** (clean: query/tools/answers fields). Single-turn (call only, no answer-narration). Glaive-function-calling-v2 is the noted scale-up for the multi-turn "feed result back → narrate" loop.
- **Tools-in-prompt → generalization:** the model learns the skill "read offered tools, pick + fill the call", not memorized tools — why a tiny model can do it at all. Important teaching point.
- **block_size 256 → 512 via RoPE, for free:** function-calling prompts are long. We load `modern.pt` rebuilt at `block_size=512` — and because Ch.4 dropped the learned position table for RoPE, **no positional param needs resizing**; the weights load unchanged and RoPE extrapolates. Validated (state_dict keys match across block_size; runs at len>trained). This is a concrete callback to Ch.4's RoPE benefit.
- Low LR 2e-5, 2 epochs (same gentle-SFT reasoning as 3b).

**Verified this session:** all checks pass with reference solutions (throwaway `validate_ch5.py`, not committed) — format/encode/collate, `parse_and_execute` (real calc=44.1, search grounds "Paris", single-dict + non-JSON handled), and the block_size-extension load test.

**Status:** authored + internally verified, **not yet run by the user** (no `toolcaller.pt`). Needs `modern.pt` from Ch.4 (exists — Ch.4 was run, val MA-loss ~3.7). Honest expectation: at 50M the win is *parseable schema-matching JSON calls*; tool choice/args will be imperfect (defensive parser handles it). Built via stdlib-json builder + cell-id pass; validated with `nbformat`.

## Dataset research (done, not yet acted on)

Researched what it'd take to train locally toward: multilingual + function-calling/agentic capability. Full writeup is in conversation history; summary:

**Reality check:** full from-scratch pretraining to get genuinely good multilingual fluency takes hundreds of billions of tokens — not realistic on one consumer GPU. Two viable paths were discussed:

1. **Fully from scratch, scoped small** — pretrain a small model on a curated multilingual slice, accept it'll be limited, then fine-tune for function calling. Good for the from-scratch learning track.
2. **Continue-pretrain / LoRA fine-tune an existing tiny open base** (e.g. SmolLM2-360M, Qwen2.5-0.5B) — much faster path to an actually-useful local agentic multilingual model, since broad language ability is already there.

Staged dataset plan (works for either path):

| Stage | Purpose | Datasets |
|---|---|---|
| 1. Pretrain / continue-pretrain | Language + multilingual ability | FineWeb-2, CulturaX, or MADLAD-400 — sample a handful of languages, a few GB each (don't download in full) |
| 2. Instruction-tuning | Make it follow instructions at all | OpenHermes-2.5, UltraChat, OASST2, Dolly-15k |
| 3. Function-calling / agentic | Teach tool-use | Glaive-function-calling-v2, xLAM/AgentOhana, APIGen-generated data, ToolBench/ToolLLM, Hermes-Function-Calling |

No dataset has been downloaded yet. No decision yet on path 1 vs path 2 — that's the first thing to nail down next session.

### Verified links

**Stage 1 — pretraining:**
- [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
- [allenai/MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400)

**Stage 2 — instruction-following:**
- [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Stage 3 — function calling / agentic:**
- [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [ToolBench (OpenBMB, GitHub)](https://github.com/OpenBMB/ToolBench)

**Path 2 candidate base models:**
- [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)

## Next step (where to pick up)

Chapter 2 proved the decoder-only architecture + training loop works end to end (English, single small dataset). **Next: scale that same pipeline toward the real goal** — multilingual + function-calling — using the datasources above. Concrete next actions, in order:

1. Decide: from-scratch-small (path 1) vs continue-pretrain-existing-base (path 2). This determines almost everything else.
2. If path 1: pick 5–15 target languages, pull small samples (a few GB each) from FineWeb-2 or MADLAD-400 via Hugging Face `datasets` (streaming, not full download).
   If path 2: pick the base model (e.g. SmolLM2-360M or Qwen2.5-0.5B), confirm it fits in 8GB VRAM with LoRA.
3. Build the tokenization/data pipeline (tokenizer choice matters more for multilingual — a BPE/SentencePiece tokenizer trained or reused across the target languages, not the GPT-2 English-only tokenizer).
4. Write the training script — use PyTorch built-in transformer layers per the decision above, not the chapter-1 from-scratch version.
5. Stage 1 (pre)training run on the multilingual slice, sanity-check generation quality.
6. Stage 2 SFT on instruction data, then Stage 3 SFT on function-calling data, each as a separate fine-tuning step on top of stage 1's checkpoint.
