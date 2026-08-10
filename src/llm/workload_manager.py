import threading
import uuid
from dataclasses import dataclass


@dataclass
class Sequence:
    # One unit of work, flowing engine -> executor -> worker and back.
    # The worker appends one token to `output` per step and flips `finished`.
    request_id: str
    prompt: str
    temperature: float = 0.7  # sampling randomness; 0 = greedy
    output: str = ""
    token_count: int = 0
    finished: bool = False


class WorkloadManager:
    def __init__(self, max_batch_size: int = 8):
        self.sequences: dict[str, Sequence] = {}
        self.max_batch_size = max_batch_size
        # Request threads add/pop while the scheduler thread batches/updates.
        self._lock = threading.Lock()

    def add_request(self, prompt: str, temperature: float = 0.7, request_id: str | None = None) -> str:
        with self._lock:
            sequence = Sequence(
                request_id=request_id or str(uuid.uuid4()),
                prompt=prompt,
                temperature=temperature,
            )
            self.sequences[sequence.request_id] = sequence
            return sequence.request_id

    def get_next_batch(self) -> list[Sequence]:
        # All unfinished sequences, oldest first, up to max_batch_size.
        # Called between every token step, so new arrivals join mid-generation.
        with self._lock:
            active = [s for s in self.sequences.values() if not s.finished]
            return active[: self.max_batch_size]

    def update_sequence(self, sequence: Sequence) -> str:
        # The worker returns pickled copies, so replace rather than mutate.
        # Returns the newly generated text (for streaming).
        with self._lock:
            old = self.sequences.get(sequence.request_id)
            if old is None:
                return ""  # dropped mid-flight (client disconnected)
            self.sequences[sequence.request_id] = sequence
            return sequence.output[len(old.output) :]

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            return self.sequences[request_id].finished

    def pop(self, request_id: str) -> Sequence | None:
        # Remove and return a sequence; each request pops its own id exactly
        # once, when it's done with it (finished, or client disconnected).
        with self._lock:
            return self.sequences.pop(request_id, None)
