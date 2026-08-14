import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    # Per-request sampling knobs, named after vLLM's SamplingParams. The
    # field defaults are the single source of truth for HTTP, CLI, and engine.
    temperature: float = 1.0  # 0 = greedy; higher flattens the distribution
    top_p: float = 1.0  # nucleus sampling: keep the top tokens with probability mass p; 1.0 = disabled
    top_k: int = (
        -1
    )  # keep only the k most likely tokens; <= 0 = disabled (vLLM's convention)
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
            raise ValueError(
                "stop strings must be non-empty"
            )  # "" would match instantly


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

    # Generated UTF-8 string
    output: str = ""

    # Generated token ids, appended by the worker;
    token_ids: list[int] = field(default_factory=list)

    # Cursors into token_ids for incremental detokenization (engine-side).
    prefix_offset: int = 0
    read_offset: int = 0
    prompt_tokens: int = 0
    token_count: int = 0

    finished: bool = False
    finish_reason: str | None = (
        None  # "stop" (EOS or a stop string) or "length" (hit max_tokens)
    )
    timing: Timing = field(default_factory=Timing)


class WorkloadManager:
    def __init__(self, max_num_seqs: int = 256):
        self.sequences: dict[str, Sequence] = {}  # request_id -> Sequence
        self.waiting: deque[str] = (
            deque()
        )  # holds not-yet-admitted requests in FCFS order
        self.running: list[str] = []  # holds the ones currently in the decode batch

        self.max_num_seqs = max_num_seqs  # cap on len(running) aka seq/batch

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
            self.waiting.append(sequence.request_id)
            return sequence.request_id

    def get_next_batch(self) -> list[Sequence]:
        # Admit waiting sequences into free running slots (FCFS), then return
        # the running set. Called between every token step, so new arrivals
        # join generation mid-flight (continuous batching).
        with self._lock:
            while self.waiting and len(self.running) < self.max_num_seqs:
                self.running.append(self.waiting.popleft())
            return [self.sequences[rid] for rid in self.running]

    def update_sequence(self, sequence: Sequence) -> str:
        # The worker returns pickled copies, so replace rather than mutate.
        # Returns the newly generated text (for streaming); "" if a
        # stop-string match shrank the output.
        with self._lock:
            old = self.sequences.get(sequence.request_id)
            if old is None:
                return ""  # dropped mid-flight (client disconnected)
            self.sequences[sequence.request_id] = sequence
            if sequence.finished:
                # 'deqeue from 'running'
                self.running.remove(sequence.request_id)
            return sequence.output[len(old.output) :]

    def pop(self, request_id: str) -> Sequence | None:
        # Remove and return a sequence, freeing its slot in whichever queue
        # holds it; each request pops its own id exactly once, when it's done
        # with it (finished, or client disconnected).
        with self._lock:
            sequence = self.sequences.pop(request_id, None)
            if sequence is not None:
                if request_id in self.running:
                    self.running.remove(request_id)
                elif not sequence.finished:
                    self.waiting.remove(request_id)
            return sequence

    def num_running(self) -> int:
        with self._lock:
            return len(self.running)

    def num_waiting(self) -> int:
        with self._lock:
            return len(self.waiting)

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            return self.sequences[request_id].finished
