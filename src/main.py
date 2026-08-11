import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm import LLMEngine
from llm.logging_config import setup_logging
from llm.workload_manager import Sequence

setup_logging()

MODEL_NAME = os.environ.get("NANO_VLLM_MODEL", "facebook/opt-125m")

engine: LLMEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model on server startup, not import (imports must stay cheap).
    global engine
    engine = LLMEngine(MODEL_NAME)
    yield


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
            {"id": MODEL_NAME, "object": "model", "created": started_at, "owned_by": "nano-vllm"}
        ],
    }


def completion_choice(index: int, text: str, finish_reason: str | None) -> dict:
    return {"index": index, "text": text, "logprobs": None, "finish_reason": finish_reason}


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    assert engine is not None
    llm = engine  # bind locally: the assert's narrowing doesn't reach the closure
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"model {request.model!r} does not exist")

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    base = {"id": completion_id, "object": "text_completion", "created": int(time.time()),
            "model": MODEL_NAME}
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
