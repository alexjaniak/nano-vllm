from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import torch

from .model_manager import ModelManager
from .model_runner import SequenceCache
from .models import get_runner
from .utils import get_device, setup_logging
from .workload_manager import SamplingParams, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DecodeState:
    # Everything the worker keeps between steps for one sequence
    cache: SequenceCache
    last_token: int = -1  # fed as the next step's input; unset during prefill


class ModelWorker:
    def __init__(self, model_name: str, dtype: str, revision: str | None = None):
        self.device = get_device()
        self.model, self.tokenizer = ModelManager().load_model(
            model_name, dtype, revision
        )
        self.model = self.model.to(self.device)
        self.runner = get_runner(self.model)
        self.states: dict[str, DecodeState] = {}  # request_id -> DecodeState

        # Per-sequence RNG for seeded requests
        self.generators: dict[str, torch.Generator] = {}

        logger.info("loaded %s on %s", model_name, self.device)

    @staticmethod
    def run(
        model_name: str,
        dtype: str,
        revision: str | None,
        task_queue: mp.Queue[list[Sequence] | None],
        result_queue: mp.Queue[list[Sequence]],
        ready_event: Event,
    ):
        # Worker process entry point: load the model once, serve steps until
        # the shutdown sentinel (None) arrives. Exiting the process is what
        # frees the model's GPU memory.
        setup_logging()
        worker = ModelWorker(model_name, dtype, revision)
        ready_event.set()  # model loaded — the server can report ready
        while (batch := task_queue.get()) is not None:
            result_queue.put(worker.forward_step(batch))
        logger.info("worker exiting")

    def forward_step(self, batch: list[Sequence]) -> list[Sequence]:
        # One new token per sequence, from a single packed forward pass.
        # Prefills contribute their whole prompt, decodes one token; the
        # ragged batch needs no padding and no prefill/decode split.
        #
        # `positions` carries each token's index within its own sequence —
        # packing destroys that (a token's buffer index means nothing), and
        # RoPE needs it. Same tensor vLLM's runner calls `positions`.
        input_chunks: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        q_lens: list[int] = []
        caches: list[SequenceCache] = []
        for sequence in batch:
            state = self.states.get(sequence.request_id)
            if state is None:
                # PREFILL: the whole prompt enters the batch at once.
                ids = self.tokenizer(sequence.prompt, return_tensors="pt").input_ids[0]
                sequence.prompt_tokens = ids.shape[0]

                state = DecodeState(cache=self.runner.new_cache())
                self.states[sequence.request_id] = state

                # Create torch generator on device for sampling
                if sequence.sampling_params.seed is not None:
                    self.generators[sequence.request_id] = torch.Generator(
                        device=self.device
                    ).manual_seed(sequence.sampling_params.seed)
            else:
                # DECODE: just the previous token; the rest is cached.
                ids = torch.tensor([state.last_token])
            start = state.cache.seq_len
            input_chunks.append(ids)
            positions.append(torch.arange(start, start + ids.shape[0]))
            q_lens.append(ids.shape[0])
            caches.append(state.cache)

        logits = self.runner.forward(
            torch.cat(input_chunks).to(self.device),
            torch.cat(positions).to(self.device),
            q_lens,
            caches,
        )

        for row, sequence in zip(logits, batch):
            state = self.states[sequence.request_id]
            token_id = self.__sample(
                row,
                sequence.sampling_params,
                self.generators.get(sequence.request_id),
            )
            # The worker's whole contract: append the sampled id. Text and
            # finish decisions (EOS/stop/length) are the engine's job.
            sequence.token_ids.append(token_id)
            sequence.token_count += 1
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
