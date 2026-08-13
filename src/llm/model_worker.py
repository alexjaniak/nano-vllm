from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
from typing import Any

import torch
from transformers import DynamicCache

from .logging_config import setup_logging
from .model_manager import ModelManager
from .workload_manager import SamplingParams, Sequence

logger = logging.getLogger(__name__)


def get_device() -> str:
    # Prefer CUDA (NVIDIA), then MPS (Apple Silicon), then fall back to CPU.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelWorker:
    def __init__(self, model_name: str, dtype: str):
        self.device = get_device()
        self.model, self.tokenizer = ModelManager().load_model(model_name, dtype)
        self.model = self.model.to(self.device)
        # Per-sequence KV cache: request_id -> (cache, next input token id).
        # The cache holds every past position's attention keys/values, so a
        # decode step only computes attention for the one new token instead
        # of re-processing the whole text — O(n) generation instead of O(n^2).
        # It lives here (not on Sequence) because cache tensors are far too
        # big to ship through the mp queues every step.
        self.states: dict[str, tuple[Any, int]] = {}
        # Per-sequence RNG for seeded requests, created once at prefill —
        # re-seeding every step would draw the same quantile forever.
        self.generators: dict[str, torch.Generator] = {}
        logger.info("loaded %s on %s", model_name, self.device)

    @staticmethod
    def run(
        model_name: str,
        dtype: str,
        task_queue: mp.Queue[list[Sequence] | None],
        result_queue: mp.Queue[list[Sequence]],
        ready_event: Event,
    ):
        # Worker process entry point: load the model once, serve steps until
        # the shutdown sentinel (None) arrives. Exiting the process is what
        # frees the model's GPU memory.
        setup_logging()
        worker = ModelWorker(model_name, dtype)
        ready_event.set()  # model loaded — the server can report ready
        while (batch := task_queue.get()) is not None:
            result_queue.put(worker.forward_step(batch))
        logger.info("worker exiting")

    def forward_step(self, batch: list[Sequence]) -> list[Sequence]:
        # One new token per sequence. Each sequence runs its own forward pass:
        # per-sequence caches have different lengths, and batching ragged
        # caches is exactly the problem paged attention solves (next).
        for sequence in batch:
            params = sequence.sampling_params
            state = self.states.get(sequence.request_id)
            if state is None:
                # Prefill: process the whole prompt once, caching every
                # position's K/V along the way.
                input_ids = self.tokenizer(
                    sequence.prompt, return_tensors="pt"
                ).input_ids.to(self.device)
                sequence.prompt_tokens = input_ids.shape[1]
                cache = DynamicCache()
                if params.seed is not None:
                    # CPU generator on purpose — see _sample.
                    self.generators[sequence.request_id] = (
                        torch.Generator().manual_seed(params.seed)
                    )
            else:
                # Decode: feed only the previous token; everything before it
                # is already in the cache.
                cache, last_token = state
                input_ids = torch.tensor([[last_token]], device=self.device)

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids, past_key_values=cache, use_cache=True
                )

            token_id = self._sample(
                output.logits[0, -1], params, self.generators.get(sequence.request_id)
            )
            sequence.token_count += 1
            if token_id == self.tokenizer.eos_token_id:
                sequence.finished = True
                sequence.finish_reason = "stop"
            else:
                sequence.output += str(self.tokenizer.decode([token_id]))
                # Stop strings: search the whole accumulated output — a stop
                # string can span token boundaries ("</s>" arriving as "</" +
                # "s>"). The match onward is cut from the output, like vLLM.
                stop_matches = [
                    index
                    for stop in params.stop or []
                    if (index := sequence.output.find(stop)) != -1
                ]
                if stop_matches:
                    sequence.output = sequence.output[: min(stop_matches)]
                    sequence.finished = True
                    sequence.finish_reason = "stop"
                elif sequence.token_count >= params.max_tokens:
                    sequence.finished = True
                    sequence.finish_reason = "length"
                else:
                    self.states[sequence.request_id] = (
                        output.past_key_values,
                        token_id,
                    )

        # Free cache + generator of anything no longer running: finished this
        # step, or dropped mid-flight (stopped appearing in batches).
        active_ids = {s.request_id for s in batch if not s.finished}
        self.states = {
            rid: state for rid, state in self.states.items() if rid in active_ids
        }
        self.generators = {
            rid: gen for rid, gen in self.generators.items() if rid in active_ids
        }
        return batch

    def _sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> int:
        # vLLM's order of operations: temperature -> top_k -> top_p -> sample.
        if params.temperature == 0:
            return int(logits.argmax())  # greedy; the other knobs are moot

        # Higher temperature flattens the distribution, lower sharpens it.
        logits = logits / params.temperature

        if params.top_k > 0:
            # Everything below the k-th best logit can't be sampled.
            kth_best = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth_best, float("-inf"))

        if params.top_p < 1.0:
            # Nucleus: cut a token when the sorted mass *before* it already
            # exceeds p (exclusive cumsum, so the best token always survives).
            sorted_logits, sorted_indices = logits.sort(descending=True)
            probs = sorted_logits.softmax(dim=-1)
            cumulative = probs.cumsum(dim=-1)
            sorted_logits[(cumulative - probs) > params.top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(
                0, sorted_indices, sorted_logits
            )

        probs = logits.softmax(dim=-1)
        if generator is not None:
            # Seeded draws run on CPU: MPS generator support is flaky, and a
            # vocab-sized copy per token is negligible next to the forward pass.
            return int(torch.multinomial(probs.float().cpu(), 1, generator=generator))
        return int(torch.multinomial(probs, num_samples=1))
