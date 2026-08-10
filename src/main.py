import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from llm import LLMEngine
from llm.logging_config import setup_logging

setup_logging()

engine: LLMEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model on server startup, not import (imports must stay cheap).
    global engine
    engine = LLMEngine()
    yield


app = FastAPI(title="nano-vllm", lifespan=lifespan)


@dataclass
class GenerateRequest:
    prompts: list[str]


@dataclass
class GenerateResponse:
    texts: list[str]


@dataclass
class StreamRequest:
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    assert engine is not None
    return GenerateResponse(texts=engine.generate(request.prompts))


@app.post("/generate_stream")
def generate_stream(request: StreamRequest):
    assert engine is not None
    llm = engine  # bind locally: the assert's narrowing doesn't reach the closure

    def event_stream():
        # Server-sent events: one "data: {...}" line per generated token.
        for token in llm.stream(request.prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
