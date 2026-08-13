import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    # Per-request sampling knobs, named after vLLM's SamplingParams. The
    # field defaults are the single source of truth for HTTP, CLI, and engine.
    temperature: float = 1.0  # 0 = greedy; higher flattens the distribution
    top_p: float = 1.0  # nucleus sampling: keep the top tokens with probability mass p; 1.0 = disabled
    top_k: int = -1  # keep only the k most likely tokens; <= 0 = disabled (vLLM's convention)
    max_tokens: int = 16  # generation cap (OpenAI's default)
    stop: str | list[str] | None = None  # stop string(s), excluded from the output
    seed: int | None = None  # per-request RNG seed for reproducible sampling

    def __post_init__(self):
        # Same guardrails as vLLM's _verify_args, for direct engine users;
        # the HTTP layer validates separately (pydantic).
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.stop is not None and any(s == "" for s in self.stop):
            raise ValueError("stop strings must be non-empty")  # "" would match instantly


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
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    output: str = ""
    prompt_tokens: int = 0
    token_count: int = 0
    finished: bool = False
    finish_reason: str | None = None  # "stop" (EOS or a stop string) or "length" (hit max_tokens)
    timing: Timing = field(default_factory=Timing)


class WorkloadManager:
    def __init__(self, max_num_seqs: int = 8):
        self.sequences: dict[str, Sequence] = {}
        self.max_num_seqs = max_num_seqs  # vLLM's name for the batch-size cap
        # Request threads add/pop while the scheduler thread batches/updates.
        self._lock = threading.Lock()

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> str:
        with self._lock:
            sequence = Sequence(
                request_id=request_id or str(uuid.uuid4()),
                prompt=prompt,
                sampling_params=sampling_params,
            )
            self.sequences[sequence.request_id] = sequence
            return sequence.request_id

    def get_next_batch(self) -> list[Sequence]:
        # All unfinished sequences, oldest first, up to max_num_seqs.
        # Called between every token step, so new arrivals join mid-generation.
        with self._lock:
            active = [s for s in self.sequences.values() if not s.finished]
            return active[: self.max_num_seqs]

    def update_sequence(self, sequence: Sequence) -> str:
        # The worker returns pickled copies, so replace rather than mutate.
        # Returns the newly generated text (for streaming); "" if a
        # stop-string match shrank the output.
        with self._lock:
            old = self.sequences.get(sequence.request_id)
            if old is None:
                return ""  # dropped mid-flight (client disconnected)
            self.sequences[sequence.request_id] = sequence
            return sequence.output[len(old.output) :]

    def pop(self, request_id: str) -> Sequence | None:
        # Remove and return a sequence; each request pops its own id exactly
        # once, when it's done with it (finished, or client disconnected).
        with self._lock:
            return self.sequences.pop(request_id, None)

    def active_count(self) -> int:
        # Unfinished sequences; those beyond max_num_seqs are "waiting".
        with self._lock:
            return sum(not s.finished for s in self.sequences.values())

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            return self.sequences[request_id].finished
