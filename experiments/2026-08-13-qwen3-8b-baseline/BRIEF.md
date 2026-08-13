# nano-vllm vs vLLM — benchmark brief

Run `20260813T020556Z` · 2026-08-13 · everything a post needs, plus the caveats
that keep it honest.

---

## Headline

**vLLM sustains 603 output tok/s. nano-vllm sustains 46 — and 46 is what it
does at every offered load tested.** 13.1× at saturation.

Three numbers that carry the whole story:

| | |
|---|---|
| nano-vllm throughput at rates 1, 2, 4, 8, ∞ | **45.9, 46.2, 45.8, 46.0, 46.0 tok/s** |
| vLLM throughput at the same rates | **127, 252, 494, 595, 603 tok/s** |
| Cost of serving 8 sequences instead of 1 | **vLLM +8%, nano-vllm +700%** |

The middle row is a saturation curve. The top row is a flat line. Eight times
the offered load, identical throughput.

---

## Setup

**Hardware** — 1× NVIDIA RTX 5090 (32 GB GDDR7, Blackwell `sm_120`), rented on
Vast.ai. Single GPU, no tensor parallelism.

**Model** — `Qwen/Qwen3-8B`, BF16, ~16 GB of weights. Dense, standard GQA
(32 query heads / 8 KV heads), text-only. Chosen deliberately: no MoE, no MLA,
no linear attention, so both engines run the same arithmetic and the comparison
isn't dominated by one exotic kernel.

**Both engines pinned identically**

| Setting | Value |
|---|---|
| `max_num_seqs` | 8 |
| `max_model_len` | 4096 |
| `gpu_memory_utilization` | 0.90 (vLLM) |
| dtype | BF16 |

`max_num_seqs=8` on *both* is the crux of the experiment — vLLM's default is
256. Capping it removes "vLLM just batches more" as an explanation and isolates
what each engine does with the *same* number of concurrent sequences.

**Load** — `vllm bench serve`, the same harness against both:

```
--dataset-name random --random-input-len 512 --random-output-len 128
--num-prompts 200 --seed 0 --request-rate {1,2,4,8,inf}
```

Fixed 512-in / 128-out removes output-length variance as a confound. Poisson
arrival (burstiness 1.0). 200 prompts per rate, 5 rates, 2 engines = 10
measured runs.

**Method** — one server at a time (they'd otherwise contend for VRAM); a warmup
run before each engine's sweep to absorb CUDA context, `torch.compile`, and
allocator growth; 45 s of quiet between every measured run so the Grafana
timeline has legible dead zones. Total wall clock 65 min. Both servers export
identically-named `vllm:*` Prometheus metrics, scraped at 1 s into one Grafana
dashboard.

**What each engine actually is**

- **vLLM** — `FLASH_ATTN` backend, CUDA graphs captured, `torch.compile` on,
  paged KV (91,392 tokens of cache), continuous batching, chunked prefill.
- **nano-vllm** — FastAPI → scheduler thread → worker process. Already has
  iteration-level continuous batching (the batch is re-formed from all
  unfinished sequences after *every* token) and a per-sequence KV cache. Model
  execution is HuggingFace `transformers` eager with `DynamicCache`.

---

## Full results

| rate | engine | out tok/s | req/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | nano-vllm | 45.9 | 0.37 | 171,571 | 320,061 | 171.9 | 344.3 | 337,351 |
| 1 | vllm | 127.1 | 0.99 | 65 | 90 | 10.4 | 27.7 | 1,613 |
| 2 | nano-vllm | 46.2 | 0.37 | 218,052 | 417,967 | 171.5 | 344.0 | 436,572 |
| 2 | vllm | 252.3 | 1.97 | 67 | 221 | 10.9 | 32.2 | 1,762 |
| 4 | nano-vllm | 45.8 | 0.37 | 226,595 | 465,117 | 171.5 | 342.2 | 483,229 |
| 4 | vllm | 493.6 | 3.86 | 133 | 1,399 | 11.2 | 33.4 | 3,037 |
| 8 | nano-vllm | 46.0 | 0.37 | 254,216 | 492,201 | 171.6 | 344.2 | 510,730 |
| 8 | vllm | 594.7 | 4.65 | 9,079 | 16,455 | 11.2 | 33.0 | 17,898 |
| ∞ | nano-vllm | 46.0 | 0.38 | 267,479 | 515,549 | 171.8 | 345.3 | 529,994 |
| ∞ | vllm | 603.3 | 4.71 | 20,570 | 40,953 | 11.2 | 40.9 | 42,406 |

All 200 requests completed in every run. Output tokens only — prompt tokens
excluded (including them would inflate every number ~4×, since the workload is
512-in / 128-out).

### Throughput ratio

| rate | 1 | 2 | 4 | 8 | ∞ |
|---|---|---|---|---|---|
| vLLM ÷ nano-vllm | 2.8× | 5.5× | 10.8× | 12.9× | **13.1×** |

**The ratio widens with load, which localizes the gap.** A gap that stayed flat
across arrival rates would be a kernel-speed gap. One that grows as load rises
is a scheduling/batching gap. (The 2.8× at rate 1 understates it — vLLM was
arrival-limited there, not capacity-limited, delivering exactly the 0.99 req/s
offered. 13.1× is the honest capacity ratio.)

---

## The mechanism

nano-vllm's scheduler is fine. Its **executor** is the problem, and it's one
line — `src/llm/model_worker.py`:

```python
for sequence in batch:      # 8 sequences = 8 separate forward passes
```

The scheduler hands the worker a batch of 8; the worker runs them one at a
time. It's a batch in name only, so nothing is shared and throughput can't
respond to load.

The ITL column measures this directly:

| | 1 sequence | 8 sequences | cost of 8× the work |
|---|---|---|---|
| vLLM | ~10.4 ms | 11.2 ms | **+8%** |
| nano-vllm | ~21.5 ms | 171.9 ms | **+700%** |

vLLM serves eight sequences for 8% more time per step because one forward pass
reads the weights once and does eight tokens' worth of matmul. nano-vllm pays
8× because it reads the weights eight times.

(nano-vllm's 21.5 ms single-forward figure is derived: 171.9 ms ÷ 8 sequential
passes. Its measured ITL is 171.9 ms at every rate because it is saturated at
8 concurrent even when only 1 req/s is offered.)

### Decomposing the 13.1×

- nano-vllm single forward ≈ **21.5 ms**; vLLM single forward ≈ **10.4 ms** →
  roughly **2× is kernels** (HF eager vs FlashAttention + CUDA graphs +
  `torch.compile`).
- The remaining **~6.5× is batching**.

Falsifiable prediction: implementing a batched forward pass should land
nano-vllm within ~2× of vLLM, with the rest being kernel quality.

### Latency

nano-vllm's TTFT (171–267 s p50) and E2E (up to 530 s p99) are pure queueing —
200 requests behind 8 slots draining at 0.37 req/s. Not a separate finding, the
same one.

---

## Caveats to state, not bury

1. **nano-vllm runs the model through HuggingFace `transformers` eager.** vLLM
   had FlashAttention, CUDA graphs, and `torch.compile`. The 13.1× is kernels
   *and* architecture, not architecture alone — hence the ~2× / ~6.5× split
   above.
2. **`max_num_seqs=8` on both.** This is why vLLM's TTFT looks bad at rates 8
   and ∞ (9 s and 20 s) — it was capped at 8 concurrent for fairness. Its
   default of 256 would look very different.
3. **Synthetic fixed-length workload** (512/128). Real traffic has variable
   output lengths, where head-of-line blocking would hurt nano-vllm *more*.
4. **No prefix caching, no speculative decoding, no quantization** on either
   side.
5. Single run per configuration — no error bars.

---

## Artifacts

- `throughput.png` — throughput vs offered load, both engines
- `sweep/*.json` — raw `vllm bench serve` output, 10 runs
- `sweep/timeline.csv` — per-phase epoch-ms windows
- `scripts/sweep.sh`, `scripts/report.py`, `scripts/chart.py` (repo root) — reproduction
- Grafana panels worth showing: **Token Throughput** (the staircase then the
  flat line) and **Scheduler State** (running pinned at 8, waiting climbing
  to ~190)

## Closing the gap

Ordered by expected impact. The first item is ~85% of the deficit.

### 1. Batched forward pass + paged KV cache — the ~6.5×

These are one project, not two. The reason the worker loops sequentially is
stated in its own comment: each sequence owns a `DynamicCache` of a different
length, and ragged caches can't be stacked into one tensor. Padding them to a
common length wastes memory proportional to the spread.

Paged attention is the answer to exactly that. Chop the KV cache into
fixed-size blocks (16 tokens), give each sequence a block table, and let the
attention kernel gather from scattered blocks — then all 8 sequences go through
**one** forward pass. `flash_attn_with_kvcache` takes a block table directly,
and it's the same kernel vLLM calls, so it stays a fair fight.

Second win, free: per-sequence contiguous caches force pre-allocating
`max_tokens` per request. Paged blocks allocate on demand, which is worth 2–4×
more concurrent sequences from the same VRAM.

**Expected:** 46 → ~300 tok/s. Throughput starts responding to offered load,
so the flat line becomes a curve.

### 2. FlashAttention instead of HF eager — the ~2×

Swap the `transformers` forward for FlashAttention-2. On a 5090 (`sm_120`) that
means FA2, not FA3 — which is what vLLM used here too, so the comparison stays
honest. This is a `ModelWorker` change, not a new dependency on vLLM: the
kernels are ordinary pip packages.

**Expected:** ~300 → ~550 tok/s, i.e. within ~10–20% of vLLM.

### 3. CUDA graphs

Capture the decode step at a few fixed batch sizes and replay. Kills Python and
kernel-launch overhead, which at 8B on a 5090 is a real fraction of a ~10 ms
step. Requires stable shapes, so it lands after paging.

**Expected:** 20–40% at small batch.

### 4. Chunked prefill — latency, not throughput

Won't move the throughput number; fixes the p99. Right now one 512-token
prefill stalls all 8 in-flight decodes because it runs in the same sequential
loop. Split prefill into token-budget chunks and mix them into decode batches.
Visible in the Grafana Token Throughput panel as generation tok/s notching down
every time prompt tok/s spikes.

**Expected:** ITL p99 collapses toward p50.

### 5. Prefix caching — workload-dependent

Hash prompt blocks, reuse KV across requests sharing a prefix. Worth a lot on
system prompts and agent loops, worth exactly nothing on this benchmark's
random prompts. Benchmark it on a workload that has shared prefixes or it will
look broken.

### Also worth fixing

`ModelExecutor.execute_batch` pickles the whole `Sequence` list — including the
accumulated `output` string — through an `mp.Queue` both ways, every token. As
output grows this is O(n²) bytes shipped, and it shows up as ITL drifting
upward over a request's lifetime. Send token ids and deltas, not accumulated
state.
