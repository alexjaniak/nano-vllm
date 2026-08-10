"""Head-to-head benchmark: nano-vllm vs vLLM.

Run one server at a time on the same machine, then run this against each:

    uv run python bench.py                 # auto-detects which server is up

Results accumulate in bench_results.json; once both backends have been
measured, a comparison scoreboard is printed. Token counts are SSE stream
events (~1 token each), so both backends are measured the same way.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "The most surprising thing about deep learning is",
    "A short story about a lighthouse keeper:",
    "The best way to learn systems programming is",
    "In the year 2050, transportation will",
    "The recipe for a perfect weekend involves",
    "Quantum computers differ from classical ones because",
    "The ocean at midnight looks like",
    "My favorite theorem in mathematics is",
]
MAX_TOKENS = 50
RESULTS_FILE = "bench_results.json"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, MAGENTA, GREEN = "\033[96m", "\033[95m", "\033[92m"


def detect_backend(base_url: str) -> tuple[str, str]:
    # vLLM speaks the OpenAI API; ours has /health. Returns (name, model).
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
            model = json.load(response)["data"][0]["id"]
            return "vllm", model
    except urllib.error.URLError:
        pass
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
        json.load(response)
        return "nano-vllm", "?"


def stream_request(base_url: str, backend: str, model: str, prompt: str) -> dict[str, float]:
    if backend == "vllm":
        path, payload = "/v1/completions", {
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": True,
        }
    else:
        path, payload = "/generate_stream", {"prompt": prompt, "temperature": 0.7}

    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_at = None
    tokens = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            payload_in = json.loads(data)
            text = (
                payload_in["token"]
                if "token" in payload_in
                else (payload_in.get("choices") or [{}])[0].get("text", "")
            )
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                tokens += 1
    end = time.perf_counter()
    return {
        "ttft": (first_token_at or end) - start,
        "tokens": tokens,
        "seconds": end - start,
    }


def run_benchmark(base_url: str, backend: str, model: str, concurrency: int) -> dict[str, float]:
    print(f"{DIM}warming up...{RESET}")
    stream_request(base_url, backend, model, "Hello")

    print(f"{DIM}single-stream: {len(PROMPTS)} prompts, one at a time...{RESET}")
    singles = [stream_request(base_url, backend, model, p) for p in PROMPTS]
    ttft = statistics.median(r["ttft"] for r in singles)
    single_tps = sum(r["tokens"] for r in singles) / sum(r["seconds"] for r in singles)

    print(f"{DIM}concurrent: {len(PROMPTS)} prompts, {concurrency} at a time...{RESET}")
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        concurrent = list(pool.map(lambda p: stream_request(base_url, backend, model, p), PROMPTS))
    wall = time.perf_counter() - wall_start
    aggregate_tps = sum(r["tokens"] for r in concurrent) / wall

    return {"ttft_ms": ttft * 1000, "single_tps": single_tps, "aggregate_tps": aggregate_tps}


def bar(value: float, best: float, color: str, width: int = 28) -> str:
    filled = max(1, round(width * value / best)) if best > 0 else 1
    return f"{color}{'▇' * filled}{RESET}"


def print_scoreboard(results: dict[str, dict[str, float]], model: str) -> None:
    nano, vllm = results["nano-vllm"], results["vllm"]
    print()
    print(f"{BOLD}  ⚔️  nano-vllm vs vLLM{RESET}  {DIM}· {model} · {MAX_TOKENS} max tokens{RESET}")
    print()
    metrics = [
        ("time to first token", "ttft_ms", "ms", True),
        ("single-stream speed", "single_tps", "tok/s", False),
        (f"aggregate throughput ({len(PROMPTS)} prompts)", "aggregate_tps", "tok/s", False),
    ]
    for label, key, unit, lower_is_better in metrics:
        n, v = nano[key], vllm[key]
        best = min(n, v) if lower_is_better else max(n, v)
        scale = max(n, v)
        ratio = (n / v) if lower_is_better else (v / n)
        print(f"  {BOLD}{label}{RESET}")
        for name, value, color in (("nano-vllm", n, CYAN), ("vllm", v, MAGENTA)):
            marker = f" {GREEN}◀{RESET}" if value == best else ""
            print(f"    {name:<10} {bar(value, scale, color)} {value:,.1f} {unit}{marker}")
        print(f"    {DIM}vLLM advantage: {ratio:,.1f}×{RESET}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm against vLLM")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=len(PROMPTS))
    args = parser.parse_args()

    backend, model = detect_backend(args.url)
    print(f"detected {BOLD}{backend}{RESET} at {args.url} {DIM}(model: {model}){RESET}")
    metrics = run_benchmark(args.url, backend, model, args.concurrency)

    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    results[backend] = metrics
    if backend == "vllm":
        results["model"] = model
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    for key, unit in (("ttft_ms", "ms"), ("single_tps", "tok/s"), ("aggregate_tps", "tok/s")):
        print(f"  {key:>14}: {metrics[key]:,.1f} {unit}")

    if "nano-vllm" in results and "vllm" in results:
        print_scoreboard(results, results.get("model", "?"))
    else:
        other = "vllm" if backend == "nano-vllm" else "nano-vllm"
        print(f"\n{DIM}saved. start the {other} server and run again for the head-to-head.{RESET}")


if __name__ == "__main__":
    main()
