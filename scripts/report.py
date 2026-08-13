#!/usr/bin/env python3
"""Turn a sweep's result JSONs into one side-by-side table.

`vllm bench serve --save-result` writes a JSON per run; sweep.sh names them
<engine>_rate<rate>.json. This pairs them up by rate so the two engines sit on
adjacent rows, and emits Markdown — the table is meant to be pasted into a
README or a post, not just read once in a terminal.

    uv run scripts/report.py                 # newest run under results/
    uv run scripts/report.py results/2026... # a specific one
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FILENAME = re.compile(r"^(?P<engine>.+)_rate(?P<rate>[^_]+)\.json$")
# vLLM has renamed metrics across versions; take whichever key is present.
ALIASES = {
    "output_throughput": ("output_throughput",),
    "request_throughput": ("request_throughput",),
    "ttft_p50": ("median_ttft_ms", "mean_ttft_ms"),
    "ttft_p99": ("p99_ttft_ms",),
    "itl_p50": ("median_itl_ms", "median_tpot_ms"),
    "itl_p99": ("p99_itl_ms", "p99_tpot_ms"),
    "e2e_p99": ("p99_e2el_ms",),
    "completed": ("completed", "num_prompts"),
}


def pick(blob: dict, field: str) -> float | None:
    for key in ALIASES[field]:
        if (value := blob.get(key)) is not None:
            return value
    return None


def rate_key(rate: str) -> float:
    # "inf" sorts last; everything else numerically.
    return float("inf") if rate == "inf" else float(rate)


def load(run_dir: Path) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(run_dir.glob("*.json")):
        match = FILENAME.match(path.name)
        if not match:
            continue  # warmup logs and anything else we didn't name
        blob = json.loads(path.read_text())
        runs.setdefault(match["rate"], {})[match["engine"]] = blob
    return runs


def fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        candidates = sorted(Path("results").glob("*/"), reverse=True)
        if not candidates:
            print("no results/ directories found", file=sys.stderr)
            return 1
        run_dir = candidates[0]

    runs = load(run_dir)
    if not runs:
        print(f"no <engine>_rate<rate>.json files in {run_dir}", file=sys.stderr)
        return 1

    engines = sorted({e for by_engine in runs.values() for e in by_engine})
    any_blob = next(iter(next(iter(runs.values())).values()))
    print(f"# {run_dir.name}\n")
    print(f"model: `{any_blob.get('model_id', '?')}`  ")
    print(f"engines: {', '.join(f'`{e}`' for e in engines)}\n")

    header = (
        "| rate | engine | out tok/s | req/s | TTFT p50 (ms) | TTFT p99 (ms) "
        "| ITL p50 (ms) | ITL p99 (ms) | E2E p99 (ms) | done |"
    )
    print(header)
    print("|" + "---|" * 10)

    speedups: list[tuple[str, float]] = []
    for rate in sorted(runs, key=rate_key):
        by_engine = runs[rate]
        for engine in engines:
            blob = by_engine.get(engine)
            if blob is None:
                print(f"| {rate} | {engine} |" + " — |" * 8)
                continue
            print(
                f"| {rate} | {engine} "
                f"| {fmt(pick(blob, 'output_throughput'))} "
                f"| {fmt(pick(blob, 'request_throughput'), 2)} "
                f"| {fmt(pick(blob, 'ttft_p50'))} "
                f"| {fmt(pick(blob, 'ttft_p99'))} "
                f"| {fmt(pick(blob, 'itl_p50'))} "
                f"| {fmt(pick(blob, 'itl_p99'))} "
                f"| {fmt(pick(blob, 'e2e_p99'))} "
                f"| {fmt(pick(blob, 'completed'), 0)} |"
            )
        # The headline number: how many times more output tokens per second
        # vLLM sustains at this arrival rate.
        if {"vllm", "nano-vllm"} <= by_engine.keys():
            fast = pick(by_engine["vllm"], "output_throughput")
            slow = pick(by_engine["nano-vllm"], "output_throughput")
            if fast and slow:
                speedups.append((rate, fast / slow))

    if speedups:
        print("\n## vLLM output throughput / nano-vllm\n")
        print("| rate | " + " | ".join(r for r, _ in speedups) + " |")
        print("|" + "---|" * (len(speedups) + 1))
        print("| ratio | " + " | ".join(f"{s:.1f}x" for _, s in speedups) + " |")
        print(
            "\nA ratio that stays flat across arrival rates is a kernel gap; "
            "one that widens with load is a scheduling gap."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
