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
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Once upon a time", "max_tokens": 50}'

# token streaming (SSE): add "stream": true
curl -N -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Once upon a time", "max_tokens": 50, "stream": true}'
```

The client streams by default; pass `--no-streaming` to wait for the full completion.

Speaking the OpenAI protocol means any standard load-test harness works out of the box — benchmark head-to-head against real vLLM with its own tool:

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B --dataset-name sharegpt --num-prompts 200 --request-rate 4
```

Default model is `Qwen/Qwen3-0.6B` (downloads on first run), on CUDA/MPS/CPU — whatever is available. Pick a different model with `--model`:

```bash
uv run src/main.py --model Qwen/Qwen2.5-1.5B-Instruct
```