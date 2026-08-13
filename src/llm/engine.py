import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from .metrics import Metrics
from .model_executor import ModelExecutor
from .workload_manager import SamplingParams, Sequence, WorkloadManager

logger = logging.getLogger(__name__)


@dataclass
class EngineArgs:
    # Server-level config, named after vLLM's EngineArgs. These field
    # defaults are the single source of truth; the CLI reads them.
    model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "auto"  # torch dtype name; "auto" keeps the checkpoint's native precision
    max_num_seqs: int = 8  # max sequences per generation-step batch (vLLM's --max-num-seqs)


class LLMEngine:
    def __init__(self, engine_args: EngineArgs):
        self.model_name = engine_args.model
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager(max_num_seqs=engine_args.max_num_seqs)
        self.metrics = Metrics(self.model_name)

        # Per-streaming-request queues: the scheduler pushes token deltas in,
        # then the finished Sequence as the end-of-stream marker.
        self.streams: dict[str, queue.Queue[str | Sequence]] = {}

        self.model_executor.setup_worker(engine_args.model, engine_args.dtype)

        # The scheduler is the only caller of the executor. It re-forms the
        # batch from all unfinished sequences after every single-token step,
        # so new requests join generation mid-flight (continuous batching).
        self.stop_event = threading.Event()
        self.scheduler = threading.Thread(target=self.__schedule_loop, daemon=True)
        self.scheduler.start()

    def __schedule_loop(self) -> None:
        while not self.stop_event.is_set():
            batch = self.workload_manager.get_next_batch()
            self.metrics.set_queue_depths(
                running=len(batch),
                waiting=self.workload_manager.active_count() - len(batch),
            )
            if not batch:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            for sequence in batch:
                if sequence.timing.scheduled is None:
                    sequence.timing.scheduled = now
                    self.metrics.observe_queue_time(now - sequence.timing.arrival)
            step_tokens = 0  # tokens the model processed this iteration
            for sequence in self.model_executor.execute_batch(
                batch
            ):  # blocking, so only good for 1 worker
                now = time.monotonic()
                delta = self.workload_manager.update_sequence(sequence)
                # Each step samples exactly one token per sequence; a
                # sequence's first step also processed its whole prompt.
                previous_token_time = sequence.timing.last_token
                sequence.timing.last_token = now
                if previous_token_time is None:
                    sequence.timing.first_token = now
                    step_tokens += sequence.prompt_tokens
                    self.metrics.observe_first_token(sequence, now)
                else:
                    step_tokens += 1
                    self.metrics.observe_next_token(now - previous_token_time)
                stream = self.streams.get(sequence.request_id)
                if stream is not None:
                    if delta:
                        stream.put(delta)
                    if sequence.finished:
                        stream.put(sequence)  # end-of-stream marker
                if sequence.finished:
                    self.metrics.observe_finished(sequence, now)
                    logger.info(
                        "finished %s: %d token(s)",
                        sequence.request_id[:8],
                        sequence.token_count,
                    )
            self.metrics.observe_iteration(step_tokens)

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams | None = None
    ) -> list[Sequence]:
        # Queue every prompt, wait for the scheduler to finish them all.
        # None means defaults (a mutable dataclass can't be a default arg).
        sampling_params = sampling_params or SamplingParams()
        request_ids = [
            self.workload_manager.add_request(prompt, sampling_params)
            for prompt in prompts
        ]
        logger.info("queued %d request(s)", len(request_ids))
        while not all(
            self.workload_manager.is_finished(request_id) for request_id in request_ids
        ):
            time.sleep(0.01)
        # Finished sequences in the same order the prompts came in.
        sequences: list[Sequence] = []
        for request_id in request_ids:
            sequence = self.workload_manager.pop(request_id)
            assert sequence is not None  # finished sequences stay until we pop them
            sequences.append(sequence)
        return sequences

    def stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> Iterator[str | Sequence]:
        # Yields text deltas, then the finished Sequence as the final item
        # (it carries finish_reason and the token counts).
        # Register the stream *before* the scheduler can see the sequence,
        # so the first tokens can't slip past us.
        request_id = str(uuid.uuid4())
        stream: queue.Queue[str | Sequence] = queue.Queue()
        self.streams[request_id] = stream
        self.workload_manager.add_request(
            prompt, sampling_params or SamplingParams(), request_id
        )
        try:
            while True:
                item = stream.get()
                yield item
                if isinstance(item, Sequence):
                    return
        finally:
            # Runs on normal completion and on client disconnect.
            del self.streams[request_id]
            self.workload_manager.pop(request_id)

    def shutdown(self) -> None:
        # Stop scheduling first so nothing races the worker's exit, then tell
        # the worker to quit — its process death releases the GPU memory.
        self.stop_event.set()
        self.scheduler.join(timeout=10)  # finishes at most one in-flight step
        self.model_executor.shutdown()

    def is_alive(self) -> bool:
        # Liveness: a dead worker process is unrecoverable.
        return self.model_executor.is_alive()

    def is_ready(self) -> bool:
        # Readiness: the worker is up and the model has finished loading.
        return self.model_executor.is_ready()
