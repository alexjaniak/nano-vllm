#!/usr/bin/env python3
"""Turn a run into a Markdown table, and append its headline to HISTORY.md.

    python3 bench/report.py                      # newest run under experiments/
    python3 bench/report.py experiments/<run>
    python3 bench/report.py <run> --no-history   # print only

HISTORY.md is the artifact that matters. One row per scenario per run, all at
the same frozen spec, so the arc of the work is a column you can read down.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HISTORY = Path(__file__).resolve().parent / "HISTORY.md"
FILENAME = re.compile(r"^(?P<engine>.+?)__(?P<scenario>.+)\.json$")

# vLLM renames metrics across versions; take whichever key is present.
ALIASES = {
    "output_throughput": ("output_throughput",),
    "request_throughput": ("request_throughput",),
    "goodput": ("request_goodput",),
    "ttft_p50": ("median_ttft_ms", "mean_ttft_ms"),
    "ttft_p99": ("p99_ttft_ms",),
    "itl_p50": ("median_itl_ms", "median_tpot_ms"),
    "itl_p99": ("p99_itl_ms", "p99_tpot_ms"),
    "e2e_p99": ("p99_e2el_ms",),
    "completed": ("completed", "num_prompts"),
}


def pick(blob: dict, field: str):
    for key in ALIASES[field]:
        if (value := blob.get(key)) is not None:
            return value
    return None


def fmt(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def newest_run() -> Path | None:
    runs = [d for d in sorted((REPO / "experiments").glob("*/")) if any(d.glob("*__*.json"))]
    return runs[-1] if runs else None


def load(run_dir: Path) -> dict[str, dict[str, dict]]:
    """-> {scenario: {engine: blob}}, in spec order where discoverable."""
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(run_dir.glob("*.json")):
        if (m := FILENAME.match(path.name)) is None:
            continue  # manifest.json and anything else we didn't name
        runs.setdefault(m["scenario"], {})[m["engine"]] = json.loads(path.read_text())
    return runs


def scenario_order(run_dir: Path) -> list[str]:
    spec = run_dir / "manifest.json"
    if not spec.exists():
        return []
    name = json.loads(spec.read_text()).get("spec_file", "spec-v1.toml")
    text = (Path(__file__).resolve().parent / name).read_text()
    return re.findall(r'^\s*name\s*=\s*"([^"]+)"', text, re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve() if args.run else newest_run()
    if run_dir is None or not run_dir.exists():
        print("no run directories found under experiments/")
        return 1

    runs = load(run_dir)
    if not runs:
        print(f"no <engine>__<scenario>.json files in {run_dir}")
        return 1

    manifest = {}
    if (mf := run_dir / "manifest.json").exists():
        manifest = json.loads(mf.read_text())

    order = scenario_order(run_dir)
    names = sorted(runs, key=lambda n: order.index(n) if n in order else 99)

    print(f"# {run_dir.name}\n")
    if manifest:
        nano, model = manifest["nano_vllm"], manifest["model"]
        tc = manifest.get("toolchain", {}).get("vllm", {})
        print(f"spec `{manifest['spec_file']}` (`{manifest['spec_sha256'][:12]}`) · "
              f"{manifest['started_utc']}  ")
        print(f"model `{model['id']}` @ `{model['revision'][:12]}` · {model['dtype']}  ")
        print(f"vLLM `{manifest['vllm']['image']}` · nano-vllm `{nano['git_describe']}` "
              f"— {nano['subject']}  ")
        print(f"{manifest['gpu']['name']} · driver {manifest['gpu']['driver']} · "
              f"torch {tc.get('torch', '?')}\n")

    print("| scenario | engine | out tok/s | req/s | goodput req/s | TTFT p50 | TTFT p99 "
          "| ITL p50 | ITL p99 | E2E p99 | done |")
    print("|" + "---|" * 11)

    headline: list[tuple[str, float, float]] = []
    for name in names:
        by_engine = runs[name]
        for engine in sorted(by_engine):
            blob = by_engine[engine]
            print(f"| {name} | {engine} "
                  f"| {fmt(pick(blob, 'output_throughput'))} "
                  f"| {fmt(pick(blob, 'request_throughput'), 2)} "
                  f"| {fmt(pick(blob, 'goodput'), 2)} "
                  f"| {fmt(pick(blob, 'ttft_p50'))} "
                  f"| {fmt(pick(blob, 'ttft_p99'))} "
                  f"| {fmt(pick(blob, 'itl_p50'))} "
                  f"| {fmt(pick(blob, 'itl_p99'))} "
                  f"| {fmt(pick(blob, 'e2e_p99'))} "
                  f"| {fmt(pick(blob, 'completed'), 0)} |")
        if {"vllm", "nano-vllm"} <= by_engine.keys():
            fast = pick(by_engine["vllm"], "output_throughput")
            slow = pick(by_engine["nano-vllm"], "output_throughput")
            if fast and slow:
                headline.append((name, fast / slow, 100 * slow / fast))

    if headline:
        # The number to track. Measured in one sweep on one box, so it is
        # immune to a different rental GPU or a driver bump.
        print("\n## Headline — vLLM ÷ nano-vllm\n")
        print("| scenario | " + " | ".join(n for n, _, _ in headline) + " |")
        print("|" + "---|" * (len(headline) + 1))
        print("| vLLM faster by | " + " | ".join(f"{r:.1f}×" for _, r, _ in headline) + " |")
        print("| nano-vllm at | " + " | ".join(f"{p:.1f}%" for _, _, p in headline) + " |")

    if headline and not args.no_history and manifest:
        append_history(manifest, runs, headline, run_dir)
        print(f"\nappended to {HISTORY.relative_to(REPO)}")
    return 0


def append_history(manifest: dict, runs: dict, headline: list, run_dir: Path) -> None:
    if not HISTORY.exists():
        HISTORY.write_text(
            "# Benchmark history\n\n"
            "Every row is the same frozen spec on one GPU, both engines measured in the\n"
            "same sweep. Only the engine changes. Rows are appended by `bench/report.py`.\n\n"
            "| date | spec | nano-vllm | scenario | nano tok/s | vLLM tok/s | vLLM faster by | nano at | run |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
    date = manifest["started_utc"][:10]
    spec = f"v{manifest['spec_version']}"
    sha = manifest["nano_vllm"]["git_describe"]
    with HISTORY.open("a") as fh:
        for name, ratio, pct in headline:
            nano = pick(runs[name]["nano-vllm"], "output_throughput")
            vllm = pick(runs[name]["vllm"], "output_throughput")
            fh.write(f"| {date} | {spec} | `{sha}` | {name} | {nano:,.1f} | {vllm:,.1f} "
                     f"| {ratio:.1f}× | {pct:.1f}% | `{run_dir.name}` |\n")


if __name__ == "__main__":
    raise SystemExit(main())
