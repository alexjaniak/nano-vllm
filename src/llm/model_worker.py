from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

import torch
from transformers import DynamicCache

from .logging_config import setup_logging
from .model_manager import ModelManager
from .workload_manager import Sequence

logger = logging.getLogger(__name__)


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
        # Per-sequence KV cache: request_id -> (cache, next input token id).
        # The cache holds every past position's attention keys/values, so a
        # decode step only computes attention for the one new token instead
        # of re-processing the whole text — O(n) generation instead of O(n^2).
        # It lives here (not on Sequence) because cache tensors are far too
        # big to ship through the mp queues every step.
        self.states: dict[str, tuple[Any, int]] = {}
        logger.info("loaded %s on %s", model_name, self.device)

    def forward_step(self, batch: list[Sequence]) -> list[Sequence]:
        # One new token per sequence. Each sequence runs its own forward pass:
        # per-sequence caches have different lengths, and batching ragged
        # caches is exactly the problem paged attention solves (next).
        for sequence in batch:
            state = self.states.get(sequence.request_id)
            if state is None:
                # Prefill: process the whole prompt once, caching every
                # position's K/V along the way.
                input_ids = self.tokenizer(sequence.prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
                sequence.prompt_tokens = input_ids.shape[1]
                cache = DynamicCache()
            else:
                # Decode: feed only the previous token; everything before it
                # is already in the cache.
                cache, last_token = state
                input_ids = torch.tensor([[last_token]], device=self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)

            # Temperature sampling: scale the logits, then draw from the
            # distribution. Higher temperature flattens it (more random);
            # 0 means greedy (always the most likely token).
            logits = output.logits[0, -1]
            if sequence.temperature > 0:
                probs = torch.softmax(logits / sequence.temperature, dim=-1)
                token_id = int(torch.multinomial(probs, num_samples=1))
            else:
                token_id = int(logits.argmax())
            sequence.token_count += 1
            if token_id == self.tokenizer.eos_token_id:
                sequence.finished = True
                sequence.finish_reason = "stop"
                self.states.pop(sequence.request_id, None)
            else:
                sequence.output += str(self.tokenizer.decode([token_id]))
                if sequence.token_count >= sequence.max_tokens:
                    sequence.finished = True
                    sequence.finish_reason = "length"
                    self.states.pop(sequence.request_id, None)
                else:
                    self.states[sequence.request_id] = (output.past_key_values, token_id)

        # A sequence that stops appearing in batches was dropped mid-flight
        # (client disconnect) — free its cache.
        batch_ids = {sequence.request_id for sequence in batch}
        self.states = {rid: state for rid, state in self.states.items() if rid in batch_ids}
        return batch

    @staticmethod
    def run(
        model_name: str,
        task_queue: mp.Queue[list[Sequence] | None],
        result_queue: mp.Queue[list[Sequence]],
    ):
        # Worker process entry point: load the model once, serve steps until
        # the shutdown sentinel (None) arrives. Exiting the process is what
        # frees the model's GPU memory.
        setup_logging()
        worker = ModelWorker(model_name)
        while (batch := task_queue.get()) is not None:
            result_queue.put(worker.forward_step(batch))
        logger.info("worker exiting")
