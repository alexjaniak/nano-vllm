from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any

import torch
from transformers import DynamicCache

from .model_manager import ModelManager
from .utils import get_device, setup_logging
from .workload_manager import SamplingParams, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DecodeState:
    # Everything the worker keeps between steps for one sequence
    cache: Any  # KV Cache
    last_token: int = -1  # fed as the next step's input; unset during prefill


class ModelWorker:
    def __init__(self, model_name: str, dtype: str):
        self.device = get_device()
        self.model, self.tokenizer = ModelManager().load_model(model_name, dtype)
        self.model = self.model.to(self.device)
        self.states: dict[str, DecodeState] = {}  # request_id -> DecodeState

        # Per-sequence RNG for seeded requests
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
        # One new token per sequence
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

                state = DecodeState(cache=DynamicCache())
                self.states[sequence.request_id] = state

                if params.seed is not None:
                    self.generators[sequence.request_id] = torch.Generator(
                        device=self.device
                    ).manual_seed(params.seed)
            else:
                # Decode: feed only the previous token; everything before it
                # is already in the cache.
                input_ids = torch.tensor([[state.last_token]], device=self.device)

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids, past_key_values=state.cache, use_cache=True
                )

            token_id = self.__sample(
                output.logits[0, -1], params, self.generators.get(sequence.request_id)
            )
            # The worker's whole contract: append the sampled id. Text and
            # finish decisions (EOS/stop/length) are the engine's job.
            sequence.token_ids.append(token_id)
            sequence.token_count += 1
            state.cache = output.past_key_values
            state.last_token = token_id

        # Free cache + generator of anything the engine stopped sending:
        # finished last step, preempted, or dropped (client disconnected).
        batch_ids = {s.request_id for s in batch}
        self.states = {
            rid: state for rid, state in self.states.items() if rid in batch_ids
        }
        self.generators = {
            rid: gen for rid, gen in self.generators.items() if rid in batch_ids
        }
        return batch

    def __sample(
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
        return int(torch.multinomial(probs, num_samples=1, generator=generator))
