import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from llm.engine import EngineArgs, LLMEngine
from llm.logging_config import setup_logging
from llm.workload_manager import SamplingParams, Sequence

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The endpoints below are sync `def`, so Starlette runs each in anyio's
    # thread pool — capped at 40 threads by default. Under a load test that
    # cap becomes the bottleneck instead of the engine: excess requests queue
    # invisibly at the ASGI layer, so num_requests_waiting under-reports and
    # hides the head-of-line blocking it exists to show.
    anyio.to_thread.current_default_thread_limiter().total_tokens = 512
    # Load the model on server startup, not import (imports must stay cheap).
    # The CLI (bottom of this file) put the EngineArgs on app.state.
    app.state.engine = LLMEngine(app.state.engine_args)
    yield
    # Server stopping: shut the worker process down so the model's GPU memory
    # is released even when uvicorn isn't killed from a terminal.
    app.state.engine.shutdown()


app = FastAPI(title="nano-vllm", lifespan=lifespan)
started_at = int(time.time())


class APIError(HTTPException):
    # HTTPException plus OpenAI's machine-readable error code.
    def __init__(self, status_code: int, message: str, code: str | None = None):
        super().__init__(status_code, message)
        self.code = code


def openai_error(status_code: int, message: str, code: str | None) -> JSONResponse:
    # OpenAI's error envelope; client SDKs pattern-match on it.
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError):
    # OpenAI clients expect 400 + the error envelope, not FastAPI's 422 shape.
    error = exc.errors()[0]
    param = ".".join(str(part) for part in error["loc"] if part != "body")
    return openai_error(400, f"{param}: {error['msg']}", code=error["type"])


@app.exception_handler(HTTPException)
async def on_http_error(_: Request, exc: HTTPException):
    return openai_error(exc.status_code, str(exc.detail), code=getattr(exc, "code", None))


# Defaults come from SamplingParams so there is exactly one source of truth.
_DEFAULTS = SamplingParams()


class CompletionRequest(BaseModel):
    # The subset of OpenAI's /v1/completions we implement; unknown fields are
    # ignored, out-of-range values get a 422 (bounds mirror vLLM's server).
    model: str
    prompt: str | list[str]
    max_tokens: int = Field(default=_DEFAULTS.max_tokens, ge=1)
    temperature: float = Field(default=_DEFAULTS.temperature, ge=0)
    top_p: float = Field(default=_DEFAULTS.top_p, gt=0, le=1)
    top_k: int = Field(default=_DEFAULTS.top_k, ge=-1)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
        )


@app.get("/health")
def health():
    # Liveness: fail only on the unrecoverable (a dead worker process), so
    # an orchestrator knows to restart us rather than just hold traffic.
    if not app.state.engine.is_alive():
        return JSONResponse(status_code=503, content={"status": "unhealthy"})
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # Readiness: no traffic until the model has finished loading.
    if not app.state.engine.is_ready():
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    # Prometheus exposition, same path vLLM serves it on. The instruments
    # live on the process-default registry (see llm/metrics.py).
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": app.state.engine_args.model, "object": "model", "created": started_at, "owned_by": "nano-vllm"}
        ],
    }


def completion_choice(index: int, text: str, finish_reason: str | None) -> dict:
    return {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason}


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    llm: LLMEngine = app.state.engine
    if request.model != app.state.engine_args.model:
        raise APIError(404, f"model {request.model!r} does not exist", code="model_not_found")

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    base = {"id": completion_id, "object": "text_completion", "created": int(time.time()),
            "model": request.model}
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    sampling_params = request.to_sampling_params()

    if request.stream:
        if len(prompts) != 1:
            raise APIError(400, "streaming supports a single prompt")

        def event_stream():
            # Server-sent events: one chunk per token delta, then a final
            # empty chunk carrying finish_reason, then [DONE].
            for item in llm.stream(prompts[0], sampling_params):
                if isinstance(item, Sequence):
                    choice = completion_choice(0, "", item.finish_reason)
                else:
                    choice = completion_choice(0, item, None)
                yield f"data: {json.dumps(base | {'choices': [choice]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    sequences = llm.generate(prompts, sampling_params)
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

# Engine option defaults come from EngineArgs, the single source of truth.
_ARGS = EngineArgs()


@cli.command()
def serve(
    model: Annotated[
        str, typer.Option(help="Hugging Face model to serve.")
    ] = _ARGS.model,
    dtype: Annotated[
        str,
        typer.Option(help="Torch dtype (e.g. float16); 'auto' keeps the checkpoint's native precision."),
    ] = _ARGS.dtype,
    max_num_seqs: Annotated[
        int,
        typer.Option(help="Max sequences batched per generation step (vLLM's --max-num-seqs)."),
    ] = _ARGS.max_num_seqs,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    # 8001 by default so vLLM can keep its default 8000 when both run for
    # the head-to-head benchmark.
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8001,
) -> None:
    """Serve MODEL over an OpenAI-compatible HTTP API."""
    app.state.engine_args = EngineArgs(model=model, dtype=dtype, max_num_seqs=max_num_seqs)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
