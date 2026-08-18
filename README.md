# nano-vllm

A tiny LLM inference server, built from scratch to learn how systems like [vLLM](https://github.com/vllm-project/vllm) work. The goal: implement interesting inference papers and primitives — batching, streaming, paged attention, and whatever comes next — in the smallest amount of readable code.

## Architecture

```
FastAPI (src/main.py)
  └── LLMEngine (src/llm/engine.py)    request lifecycle
        ├── WorkloadManager            queues sequences, forms FIFO batches
        └── ModelExecutor              owns the worker process + task/result queues
              └── ModelWorker          separate process: loads the model, runs batched generate()
```

One `Sequence` dataclass flows through the whole pipeline; the worker fills in its `output`.

## Run it

```bash
uv sync
uv run src/main.py
```

Every runtime option is a CLI flag — see `uv run src/main.py --help`.

Then either use the interactive client:

```bash
uv run src/client.py
```

or hit the OpenAI-compatible API directly:

```bash
# completions — `prompt` also takes a list, and the batch shares forward passes
curl -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Once upon a time", "max_tokens": 50}'

# token streaming (SSE): add "stream": true
curl -N -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Once upon a time", "max_tokens": 50, "stream": true}'

# sampling params, vLLM-style: top_p/top_k filtering, stop strings, seeded RNG
curl -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Once upon a time", "max_tokens": 50,
       "temperature": 0.8, "top_p": 0.9, "top_k": 40, "stop": ["."], "seed": 42}'
```

The client streams by default; pass `--no-streaming` to wait for the full completion.

Speaking the OpenAI protocol means any standard load-test harness works out of the box — benchmark head-to-head against real vLLM with its own tool:

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen3-0.6B --dataset-name sharegpt --num-prompts 200 --request-rate 4
```

## Benchmarks

`bench/` runs the real head-to-head on a frozen spec — same workload, same
model commit, same pinned vLLM image, every time, so runs months apart stay
comparable. The tracked number is nano-vllm's throughput as a fraction of
vLLM's, per scenario, in `bench/HISTORY.md`.

```bash
python3 bench/run.py --dry-run   # the plan and the pins
python3 bench/run.py             # full sweep (needs a GPU box + docker)
python3 bench/report.py          # table, and one row appended to HISTORY.md
```

See [`bench/README.md`](bench/README.md). Earlier results live under
`experiments/` — the v0 baseline (2026-08-13) predates the frozen spec and
isn't exactly reproducible, so v1 re-runs its workload as the `fixed` scenario
to bridge the two.

Default model is `Qwen/Qwen3-0.6B` (downloads on first run), on CUDA/MPS/CPU — whatever is available. Pick a different model with `--model`, and cap the per-step batch with `--max-num-seqs` (vLLM's flag name):

```bash
uv run src/main.py --model Qwen/Qwen2.5-1.5B-Instruct --max-num-seqs 16
```