import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm import LLMEngine
from llm.logging_config import setup_logging
from llm.workload_manager import Sequence

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on server startup, not import (imports must stay cheap).
    # The CLI (bottom of this file) put model and dtype on app.state.
    app.state.engine = LLMEngine(app.state.model, app.state.dtype)
    yield
    # Server stopping: shut the worker process down so the model's GPU memory
    # is released even when uvicorn isn't killed from a terminal.
    app.state.engine.shutdown()


app = FastAPI(title="nano-vllm", lifespan=lifespan)
started_at = int(time.time())


@dataclass
class CompletionRequest:
    # The subset of OpenAI's /v1/completions we implement; unknown fields
    # sent by clients are ignored rather than rejected.
    model: str
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float = 1.0
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": app.state.model, "object": "model", "created": started_at, "owned_by": "nano-vllm"}
        ],
    }


def completion_choice(index: int, text: str, finish_reason: str | None) -> dict:
    return {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason}


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    llm: LLMEngine = app.state.engine
    if request.model != app.state.model:
        raise HTTPException(status_code=404, detail=f"model {request.model!r} does not exist")

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    base = {"id": completion_id, "object": "text_completion", "created": int(time.time()),
            "model": app.state.model}
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    if request.stream:
        if len(prompts) != 1:
            raise HTTPException(status_code=400, detail="streaming supports a single prompt")

        def event_stream():
            # Server-sent events: one chunk per token delta, then a final
            # empty chunk carrying finish_reason, then [DONE].
            for item in llm.stream(prompts[0], request.temperature, request.max_tokens):
                if isinstance(item, Sequence):
                    choice = completion_choice(0, "", item.finish_reason)
                else:
                    choice = completion_choice(0, item, None)
                yield f"data: {json.dumps(base | {'choices': [choice]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    sequences = llm.generate(prompts, request.temperature, request.max_tokens)
    prompt_tokens = sum(s.prompt_tokens for s in sequences)
    completion_tokens = sum(s.token_count for s in sequences)
    return base | {
        "choices": [
            completion_choice(i, s.output, s.finish_reason) for i, s in enumerate(sequences)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# CLI — every runtime option lives here; `--help` lists them all.
# ---------------------------------------------------------------------------

cli = typer.Typer(add_completion=False)


@cli.command()
def serve(
    model: Annotated[
        str, typer.Option(help="Hugging Face model to serve.")
    ] = "Qwen/Qwen3-0.6B",
    dtype: Annotated[
        str,
        typer.Option(help="Torch dtype (e.g. float16); 'auto' keeps the checkpoint's native precision."),
    ] = "auto",
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Serve MODEL over an OpenAI-compatible HTTP API."""
    app.state.model, app.state.dtype = model, dtype
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
