import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator

from .model_executor import ModelExecutor
from .workload_manager import WorkloadManager

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, model_name: str = "facebook/opt-125m"):
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager()
        # Per-streaming-request queues the scheduler pushes new tokens into.
        self.streams: dict[str, queue.Queue[str | None]] = {}

        self.model_executor.setup_worker(model_name)

        # The scheduler is the only caller of the executor. It re-forms the
        # batch from all unfinished sequences after every single-token step,
        # so new requests join generation mid-flight (continuous batching).
        self.scheduler = threading.Thread(target=self._schedule_loop, daemon=True)
        self.scheduler.start()

    def _schedule_loop(self) -> None:
        while True:
            batch = self.workload_manager.get_next_batch()
            if not batch:
                time.sleep(0.01)
                continue
            for sequence in self.model_executor.execute_batch(batch):
                delta = self.workload_manager.update_sequence(sequence)
                stream = self.streams.get(sequence.request_id)
                if stream is not None:
                    if delta:
                        stream.put(delta)
                    if sequence.finished:
                        stream.put(None)  # end-of-stream marker
                if sequence.finished:
                    logger.info(
                        "finished %s: %d token(s)", sequence.request_id[:8], sequence.token_count
                    )

    def generate(self, prompts: list[str]) -> list[str]:
        # Queue every prompt, wait for the scheduler to finish them all.
        request_ids = [self.workload_manager.add_request(prompt) for prompt in prompts]
        logger.info("queued %d request(s)", len(request_ids))
        while not all(self.workload_manager.is_finished(request_id) for request_id in request_ids):
            time.sleep(0.01)
        # Outputs in the same order the prompts came in.
        outputs: list[str] = []
        for request_id in request_ids:
            sequence = self.workload_manager.pop(request_id)
            assert sequence is not None  # finished sequences stay until we pop them
            outputs.append(sequence.output)
        return outputs

    def stream(self, prompt: str) -> Iterator[str]:
        # Register the stream *before* the scheduler can see the sequence,
        # so the first tokens can't slip past us.
        request_id = str(uuid.uuid4())
        stream: queue.Queue[str | None] = queue.Queue()
        self.streams[request_id] = stream
        self.workload_manager.add_request(prompt, request_id)
        try:
            while (token := stream.get()) is not None:
                yield token
        finally:
            # Runs on normal completion and on client disconnect.
            del self.streams[request_id]
            self.workload_manager.pop(request_id)
