import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Timing:
    # Lifecycle timestamps (monotonic clock), set by the engine and carried
    # through the worker round-trip; the Prometheus metrics derive queue time,
    # TTFT, inter-token latency, and the prefill/decode split from them.
    arrival: float = field(default_factory=time.monotonic)
    scheduled: float | None = None  # first time it entered a batch
    first_token: float | None = None
    last_token: float | None = None


@dataclass
class Sequence:
    # One unit of work, flowing engine -> executor -> worker and back.
    # The worker appends one token to `output` per step and flips `finished`.
    request_id: str
    prompt: str
    temperature: float = 1.0  # sampling randomness; 0 = greedy
    max_tokens: int = 16  # generation cap (OpenAI's default)
    output: str = ""
    prompt_tokens: int = 0
    token_count: int = 0
    finished: bool = False
    finish_reason: str | None = None  # "stop" (EOS) or "length" (hit max_tokens)
    timing: Timing = field(default_factory=Timing)


class WorkloadManager:
    def __init__(self, max_batch_size: int = 8):
        self.sequences: dict[str, Sequence] = {}
        self.max_batch_size = max_batch_size
        # Request threads add/pop while the scheduler thread batches/updates.
        self._lock = threading.Lock()

    def add_request(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 16,
        request_id: str | None = None,
    ) -> str:
        with self._lock:
            sequence = Sequence(
                request_id=request_id or str(uuid.uuid4()),
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
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

    def active_count(self) -> int:
        # Unfinished sequences; those beyond max_batch_size are "waiting".
        with self._lock:
            return sum(not s.finished for s in self.sequences.values())

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            return self.sequences[request_id].finished

    def pop(self, request_id: str) -> Sequence | None:
        # Remove and return a sequence; each request pops its own id exactly
        # once, when it's done with it (finished, or client disconnected).
        with self._lock:
            return self.sequences.pop(request_id, None)
