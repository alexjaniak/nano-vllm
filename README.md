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

or hit the API directly:

```bash
# prompts are batched into one forward pass
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' -d '{"prompts": ["Once upon a time"]}'
```

Default model is `facebook/opt-125m` (downloads on first run), on CUDA/MPS/CPU — whatever is available.

## Roadmap

- [x] Model worker in a separate process
- [x] Batched generation with request correlation
- [ ] Token streaming (SSE)
- [ ] Continuous batching
- [ ] KV-cache / paged attention
- [ ] Multi-model serving
