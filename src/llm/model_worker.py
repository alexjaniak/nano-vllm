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
        logger.info("loaded %s on %s", model_name, self.device)

    def generate(self, batch: list[Sequence]) -> list[Sequence]:
        # Pad prompts to equal length so the batch forms one rectangular tensor.
        encoded = self.tokenizer(
            [sequence.prompt for sequence in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                # Rows that hit EOS early are filled with this token while the
                # rest of the batch keeps generating.
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Count real generated tokens: everything past the prompt, minus the
        # EOS fill that pads out rows that finished early.
        generated = output[:, encoded.input_ids.shape[1] :]
        token_count = int((generated != self.tokenizer.eos_token_id).sum())
        logger.info("batch of %d sequence(s): generated %d token(s)", len(batch), token_count)

        texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        for sequence, text in zip(batch, texts):
            sequence.output = text
            sequence.finished = True
        return batch

    @staticmethod
    def run(
        model_name: str,
        task_queue: mp.Queue[list[Sequence]],
        result_queue: mp.Queue[list[Sequence]],
    ):
        # Worker process entry point: load the model once, serve batches forever.
        setup_logging()
        worker = ModelWorker(model_name)
        while True:
            batch = task_queue.get()
            result_queue.put(worker.generate(batch))
