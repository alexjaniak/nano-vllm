"""Prometheus metrics, mirroring vLLM's metric names exactly.

vLLM's example Grafana dashboard queries metrics named vllm:*, so emitting
the same names, labels, and bucket boundaries lets one dashboard graph both
servers side by side, distinguished only by the Prometheus job label.

vLLM metrics we deliberately don't emit (an empty panel is honest — it shows
the feature doesn't exist here yet):
- vllm:kv_cache_usage_perc — no fixed-size cache pool to be a fraction of.
- vllm:prefix_cache_queries / vllm:prefix_cache_hits — no prefix caching.
"""

from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics

from .workload_manager import Sequence

# vLLM omits the _created companion series; match its exposition exactly.
disable_created_metrics()


def _1_2_5_buckets(limit: int) -> list[int]:
    # 1, 2, 5, 10, 20, 50, ... up to limit (vLLM's build_1_2_5_buckets).
    buckets: list[int] = []
    exponent = 0
    while True:
        for mantissa in (1, 2, 5):
            value = mantissa * 10**exponent
            if value > limit:
                return buckets
            buckets.append(value)
        exponent += 1


# Bucket boundaries copied from vLLM so histogram series line up bucket-for-
# bucket; quantiles computed from them are comparable across the two servers.
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0,
                640.0, 2560.0]
ITL_BUCKETS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
               0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]
REQUEST_TIME_BUCKETS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0,
                        20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
                        960.0, 1920.0, 7680.0]
ITERATION_TOKENS_BUCKETS = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                            4096, 8192]
# vLLM sizes these from max_model_len; a generous constant serves every model
# we're likely to load (extra empty buckets are harmless).
TOKEN_COUNT_BUCKETS = _1_2_5_buckets(8192)


class Metrics:
    """All instruments pre-bound to model_name, the label every vLLM metric
    carries. Uses the process-default registry; create once per process."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        by_model = ["model_name"]

        self.num_requests_running = Gauge(
            "vllm:num_requests_running",
            "Number of requests currently running on GPU.", by_model,
        ).labels(model_name)
        self.num_requests_waiting = Gauge(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed.", by_model,
        ).labels(model_name)

        self.prompt_tokens = Counter(
            "vllm:prompt_tokens_total",
            "Number of prefill tokens processed.", by_model,
        ).labels(model_name)
        self.generation_tokens = Counter(
            "vllm:generation_tokens_total",
            "Number of generation tokens processed.", by_model,
        ).labels(model_name)
        self.request_success = Counter(
            "vllm:request_success_total",
            "Count of successfully processed requests.",
            ["model_name", "finished_reason"],
        )
        # Pre-create the known label children so the series exist at zero
        # (rate() needs a starting sample), as vLLM does.
        for reason in ("stop", "length", "abort"):
            self.request_success.labels(model_name, reason)

        self.time_to_first_token = Histogram(
            "vllm:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            by_model, buckets=TTFT_BUCKETS,
        ).labels(model_name)
        # vLLM renamed time_per_output_token to inter_token_latency; emit both
        # so dashboards built against either version work.
        self.time_per_output_token = Histogram(
            "vllm:time_per_output_token_seconds",
            "Histogram of time per output token in seconds.",
            by_model, buckets=ITL_BUCKETS,
        ).labels(model_name)
        self.inter_token_latency = Histogram(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter-token latency in seconds.",
            by_model, buckets=ITL_BUCKETS,
        ).labels(model_name)
        self.e2e_request_latency = Histogram(
            "vllm:e2e_request_latency_seconds",
            "Histogram of end to end request latency in seconds.",
            by_model, buckets=REQUEST_TIME_BUCKETS,
        ).labels(model_name)
        self.request_queue_time = Histogram(
            "vllm:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request.",
            by_model, buckets=REQUEST_TIME_BUCKETS,
        ).labels(model_name)
        self.request_prefill_time = Histogram(
            "vllm:request_prefill_time_seconds",
            "Histogram of time spent in PREFILL phase for request.",
            by_model, buckets=REQUEST_TIME_BUCKETS,
        ).labels(model_name)
        self.request_decode_time = Histogram(
            "vllm:request_decode_time_seconds",
            "Histogram of time spent in DECODE phase for request.",
            by_model, buckets=REQUEST_TIME_BUCKETS,
        ).labels(model_name)
        self.request_inference_time = Histogram(
            "vllm:request_inference_time_seconds",
            "Histogram of time spent in RUNNING phase for request.",
            by_model, buckets=REQUEST_TIME_BUCKETS,
        ).labels(model_name)

        self.request_prompt_tokens = Histogram(
            "vllm:request_prompt_tokens",
            "Number of prefill tokens processed.",
            by_model, buckets=TOKEN_COUNT_BUCKETS,
        ).labels(model_name)
        self.request_generation_tokens = Histogram(
            "vllm:request_generation_tokens",
            "Number of generation tokens processed.",
            by_model, buckets=TOKEN_COUNT_BUCKETS,
        ).labels(model_name)
        self.iteration_tokens = Histogram(
            "vllm:iteration_tokens_total",
            "Histogram of number of tokens per engine step.",
            by_model, buckets=ITERATION_TOKENS_BUCKETS,
        ).labels(model_name)

    def set_queue_depths(self, running: int, waiting: int) -> None:
        self.num_requests_running.set(running)
        self.num_requests_waiting.set(waiting)

    def observe_queue_time(self, seconds: float) -> None:
        self.request_queue_time.observe(seconds)

    def observe_first_token(self, sequence: Sequence, now: float) -> None:
        # The step that produced a sequence's first token also prefilled its
        # whole prompt, so this is where prompt tokens are counted.
        self.prompt_tokens.inc(sequence.prompt_tokens)
        self.generation_tokens.inc()
        self.time_to_first_token.observe(now - sequence.timing.arrival)

    def observe_next_token(self, seconds_since_last: float) -> None:
        self.generation_tokens.inc()
        self.time_per_output_token.observe(seconds_since_last)
        self.inter_token_latency.observe(seconds_since_last)

    def observe_iteration(self, tokens_processed: int) -> None:
        self.iteration_tokens.observe(tokens_processed)

    def observe_finished(self, sequence: Sequence, now: float) -> None:
        self.request_success.labels(
            model_name=self.model_name,
            finished_reason=sequence.finish_reason or "abort",
        ).inc()
        timing = sequence.timing
        self.e2e_request_latency.observe(now - timing.arrival)
        if timing.scheduled is not None and timing.first_token is not None:
            self.request_prefill_time.observe(timing.first_token - timing.scheduled)
            self.request_decode_time.observe(now - timing.first_token)
            self.request_inference_time.observe(now - timing.scheduled)
        self.request_prompt_tokens.observe(sequence.prompt_tokens)
        self.request_generation_tokens.observe(sequence.token_count)
