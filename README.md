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
cd src && uv run uvicorn main:app
```

Then either use the interactive client:

```bash
python src/client.py
```

or hit the OpenAI-compatible API directly:

```bash
# completions — `prompt` also takes a list, and the batch shares forward passes
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "facebook/opt-125m", "prompt": "Once upon a time", "max_tokens": 50}'

# token streaming (SSE): add "stream": true
curl -N -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "facebook/opt-125m", "prompt": "Once upon a time", "max_tokens": 50, "stream": true}'
```

The client streams by default; pass `--no-streaming` to wait for the full completion.

Speaking the OpenAI protocol means any standard load-test harness works out of the box — benchmark head-to-head against real vLLM with its own tool:

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --model facebook/opt-125m --dataset-name sharegpt --num-prompts 200 --request-rate 4
```

Default model is `facebook/opt-125m` (downloads on first run), on CUDA/MPS/CPU — whatever is available. Pick a different model with the `NANO_VLLM_MODEL` env var:

```bash
NANO_VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct uv run uvicorn main:app
```

Models load in their checkpoint's native precision (fp16/bf16), so a ~1.5B model fits comfortably in 6 GB of VRAM.

## Roadmap

- [x] Model worker in a separate process
- [x] Batched generation with request correlation
- [x] Token streaming (SSE)
- [x] Continuous batching (token-level scheduler; new requests join mid-generation)
- [x] KV cache (per-sequence; prefill once, then one token per forward pass)
- [x] OpenAI-compatible API (`/v1/completions`, `/v1/models`)
- [ ] Paged attention (batch ragged caches back together)
- [ ] Multi-model serving
