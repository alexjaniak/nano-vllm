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

## Run it with Docker

`docker/Dockerfile` is self-contained: a CUDA runtime base plus this repo's `uv.lock`, so the image is the lockfile and nothing else. The host supplies the driver and the [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit); torch brings its own CUDA libs. Server flags are appended after the tag, the way vLLM's own image works:

```bash
docker run --gpus all -p 8001:8001 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/alexjaniak/nano-vllm:<sha> \
  --model Qwen/Qwen3-8B --max-num-seqs 8
```

Every flag is also a `NANO_*` env var — `NANO_MODEL`, `NANO_MAX_NUM_SEQS`, `NANO_HOST`, `NANO_PORT` — for orchestrators that would rather set environment than rewrite a command. A flag still wins over the env var:

```bash
docker run --gpus all -p 9000:9000 --ipc=host \
  -e NANO_MODEL=Qwen/Qwen3-8B -e NANO_PORT=9000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/alexjaniak/nano-vllm:<sha>
```

The image binds `0.0.0.0` (a container on `127.0.0.1` is unreachable through `-p`) and ships a `HEALTHCHECK` on `/ready`, which stays failing until the model is loaded. `--ipc=host` is not optional: the worker runs in its own process and the default 64MB of shared memory is not enough for it.

The build context is the repo root, and the tag is a git SHA rather than `latest` — the same reason `spec-v1.toml` pins a model commit, so an old build stays comparable to the numbers it produced:

```bash
TAG=ghcr.io/alexjaniak/nano-vllm:$(git rev-parse --short HEAD)
docker build -f docker/Dockerfile -t "$TAG" .
docker push "$TAG"
```

## Benchmarks

`bench/` runs the real head-to-head on a frozen spec — same workload, same
model commit, same pinned vLLM image, every time, so runs months apart stay
comparable. Both engines run the same image they ship as: vLLM's, and the one
above. The tracked number is nano-vllm's throughput as a fraction of
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