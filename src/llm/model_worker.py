from __future__ import annotations

import logging
import multiprocessing as mp

import torch

from .logging_config import setup_logging
from .model_manager import ModelManager
from .workload_manager import Sequence

logger = logging.getLogger(__name__)

# Cap on generated tokens per request (prompt length not counted).
MAX_NEW_TOKENS = 50


def get_device() -> str:
    # Prefer CUDA (NVIDIA), then MPS (Apple Silicon), then fall back to CPU.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelWorker:
    def __init__(self, model_name: str):
        self.device = get_device()
        self.model, self.tokenizer = ModelManager().load_model(model_name)
        self.model = self.model.to(self.device)
        # Causal LMs are trained without padding, so many ship without a pad
        # token. Reuse EOS: padded positions are masked out of attention, so
        # the id just has to exist.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so every row's *last* position is real text; logits[:, -1]
        # is then the next-token prediction for the whole batch at once.
        self.tokenizer.padding_side = "left"
        logger.info("loaded %s on %s", model_name, self.device)

    def forward_step(self, batch: list[Sequence]) -> list[Sequence]:
        # One forward pass, one new token per sequence. Re-tokenizing the full
        # text every step is O(n^2) — the KV cache will fix this later.
        encoded = self.tokenizer(
            [sequence.prompt + sequence.output for sequence in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        # Greedy decoding: most likely token at the final position of each row.
        next_tokens = output.logits[:, -1, :].argmax(dim=-1).tolist()
        for sequence, token_id in zip(batch, next_tokens):
            sequence.token_count += 1
            if token_id == self.tokenizer.eos_token_id or sequence.token_count >= MAX_NEW_TOKENS:
                sequence.finished = True
            if token_id != self.tokenizer.eos_token_id:
                sequence.output += str(self.tokenizer.decode([token_id]))
        return batch

    @staticmethod
    def run(
        model_name: str,
        task_queue: mp.Queue[list[Sequence]],
        result_queue: mp.Queue[list[Sequence]],
    ):
        # Worker process entry point: load the model once, serve steps forever.
        setup_logging()
        worker = ModelWorker(model_name)
        while True:
            batch = task_queue.get()
            result_queue.put(worker.forward_step(batch))
