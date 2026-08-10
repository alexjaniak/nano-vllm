from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

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
    prompt: str


@dataclass
class GenerateResponse:
    text: str


@dataclass
class BatchGenerateRequest:
    prompts: list[str]


@dataclass
class BatchGenerateResponse:
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/basic_generate", response_model=GenerateResponse)
def basic_generate(request: GenerateRequest):
    assert engine is not None
    return GenerateResponse(text=engine.generate([request.prompt])[0])


@app.post("/generate", response_model=BatchGenerateResponse)
def generate(request: BatchGenerateRequest):
    assert engine is not None
    return BatchGenerateResponse(texts=engine.generate(request.prompts))
