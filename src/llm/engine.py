import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from transformers import AutoTokenizer

from .metrics import Metrics
from .model_executor import ModelExecutor
from .workload_manager import SamplingParams, Sequence, WorkloadManager

logger = logging.getLogger(__name__)


@dataclass
class EngineArgs:
    # Server-level config, named after vLLM's EngineArgs. These field
    # defaults are the single source of truth; the CLI reads them.
    model: str = "Qwen/Qwen3-0.6B"
    dtype: str = (
        "auto"  # torch dtype name; "auto" keeps the checkpoint's native precision
    )
    max_num_seqs: int = (
        256  # max sequences per generation-step batch (vLLM's --max-num-seqs)
    )


class LLMEngine:
    def __init__(self, engine_args: EngineArgs):
        self.model_name = engine_args.model
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager(max_num_seqs=engine_args.max_num_seqs)
        self.metrics = Metrics(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(engine_args.model)

        # Per-request queues: the scheduler pushes token deltas in, then the
        # finished Sequence as the end-of-stream marker.
        self.streams: dict[str, asyncio.Queue[str | Sequence]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None  # set by stream()

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
                waiting=self.workload_manager.num_waiting(),
            )

            if not batch:
                self.workload_manager.new_work.wait()  # add_request or shutdown
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

                self.__process_output(sequence)
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
                    assert self.loop is not None  # stream() set it first
                    # Hop to the event loop; never block the scheduler on it.
                    if delta:
                        self.loop.call_soon_threadsafe(stream.put_nowait, delta)
                    if sequence.finished:
                        self.loop.call_soon_threadsafe(stream.put_nowait, sequence)
                if sequence.finished:
                    self.metrics.observe_finished(sequence, now)
                    logger.info(
                        "finished %s: %d token(s)",
                        sequence.request_id[:8],
                        sequence.token_count,
                    )
            self.metrics.observe_iteration(step_tokens)

    def __process_output(self, sequence: Sequence) -> None:
        # Turn the worker's raw sampled token into text and a finish decision.
        # The worker only appends to token_ids
        params = sequence.sampling_params
        if sequence.token_ids[-1] == self.tokenizer.eos_token_id:
            sequence.finished = True
            sequence.finish_reason = "stop"
            return
        previous_len = len(sequence.output)
        sequence.output += self.__detokenize(sequence)
        new_chars = len(sequence.output) - previous_len

        # Search the new text, with len(stop)-1 chars of overlap to catch
        # matches spanning token boundaries ("</s>" arriving as "</" + "s>").
        # TODO: streaming can leak a partial stop string before the match
        # completes; vLLM holds deltas back by max(len(stop)) - 1 chars.
        stop_matches = [
            index
            for stop in params.stop or []
            if (index := sequence.output.find(stop, 1 - new_chars - len(stop))) != -1
        ]
        if stop_matches:
            sequence.output = sequence.output[: min(stop_matches)]
            sequence.finished = True
            sequence.finish_reason = "stop"
        elif sequence.token_count >= params.max_tokens:
            sequence.finished = True
            sequence.finish_reason = "length"

    def __detokenize(self, sequence: Sequence) -> str:
        # Incremental detokenization: the text the newest token adds.
        #
        # Decoding one id at a time is wrong. A BPE token can be half a UTF-8
        # character, so decode([id]) returns U+FFFD for anything non-ASCII,
        # and tokenizers strip the leading-space marker off an isolated token.
        # Instead decode a short suffix twice — with and without the newest
        # token — and take the difference. Both decodes start at the same
        # offset, so a partial character at the front renders identically in
        # each and cancels out in the slice. The cursors keep the decoded
        # window a couple of tokens wide, so this stays O(1) per step.
        ids = sequence.token_ids
        prefix = self.tokenizer.decode(
            ids[sequence.prefix_offset : sequence.read_offset],
            skip_special_tokens=True,
        )
        whole = self.tokenizer.decode(
            ids[sequence.prefix_offset :], skip_special_tokens=True
        )
        # A trailing U+FFFD means the newest token opened a multi-byte
        # character that hasn't finished arriving. Hold the text back and let
        # the next token complete it, rather than emitting a replacement char.
        if len(whole) <= len(prefix) or whole.endswith("�"):
            return ""
        sequence.prefix_offset = sequence.read_offset
        sequence.read_offset = len(ids)
        return whole[len(prefix) :]

    async def generate(
        self, prompts: list[str], sampling_params: SamplingParams | None = None
    ) -> list[Sequence]:
        # Every request is a stream under the hood; the finished Sequence is
        # always the last item. gather keeps results in prompt order.
        logger.info("queued %d request(s)", len(prompts))

        async def collect(prompt: str) -> Sequence:
            last: str | Sequence | None = None
            async for item in self.stream(prompt, sampling_params):
                last = item  # exhaust fully so stream()'s cleanup runs
            assert isinstance(last, Sequence)  # the terminal item
            return last

        return list(await asyncio.gather(*(collect(prompt) for prompt in prompts)))

    async def stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[str | Sequence]:
        # Yields text deltas, then the finished Sequence as the final item
        # (it carries finish_reason and the token counts).
        # Register the stream *before* the scheduler can see the sequence,
        # so the first tokens can't slip past us.
        self.loop = asyncio.get_running_loop()
        request_id = str(uuid.uuid4())
        stream: asyncio.Queue[str | Sequence] = asyncio.Queue()
        self.streams[request_id] = stream
        self.workload_manager.add_request(
            prompt, sampling_params or SamplingParams(), request_id
        )
        try:
            while True:
                item = await stream.get()
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
        self.workload_manager.new_work.set()  # unpark the idle wait
        self.scheduler.join(timeout=10)  # finishes at most one in-flight step
        self.model_executor.shutdown()

    def is_alive(self) -> bool:
        # Liveness: a dead worker process is unrecoverable.
        return self.model_executor.is_alive()

    def is_ready(self) -> bool:
        # Readiness: the worker is up and the model has finished loading.
        return self.model_executor.is_ready()
