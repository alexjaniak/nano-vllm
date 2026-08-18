#!/usr/bin/env python3
"""Two charts: this run's per-scenario throughput, and the arc across runs.

    uv run --with matplotlib bench/chart.py                  # newest run
    uv run --with matplotlib bench/chart.py experiments/<run>

The second panel is the one worth posting — it reads HISTORY.md, so it only
becomes interesting once there are two or more runs at the same spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import HISTORY, load, newest_run, pick, scenario_order  # noqa: E402

NANO, VLLM = "#2563eb", "#94a3b8"


def history_rows() -> list[dict]:
    if not HISTORY.exists():
        return []
    rows = []
    for line in HISTORY.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 9 or cells[0] in ("date", "---") or set(cells[0]) <= {"-"}:
            continue
        try:
            rows.append({"date": cells[0], "sha": cells[2].strip("`"), "scenario": cells[3],
                         "nano": float(cells[4].replace(",", "")),
                         "vllm": float(cells[5].replace(",", "")),
                         "pct": float(cells[7].rstrip("%"))})
        except ValueError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve() if args.run else newest_run()
    if run_dir is None:
        print("no run directories found")
        return 1
    runs = load(run_dir)
    order = scenario_order(run_dir)
    names = sorted(runs, key=lambda n: order.index(n) if n in order else 99)

    rows = history_rows()
    fig, axes = plt.subplots(1, 2 if rows else 1, figsize=(13 if rows else 7, 4.6))
    ax = axes[0] if rows else axes

    # Panel 1 — this run, side by side. Log scale: at native concurrency vLLM
    # is 45x nano, and on a linear axis every nano bar collapses to nothing.
    x = range(len(names))
    nano = [pick(runs[n].get("nano-vllm", {}), "output_throughput") or 0 for n in names]
    vllm = [pick(runs[n].get("vllm", {}), "output_throughput") or 0 for n in names]
    ax.bar([i - 0.2 for i in x], vllm, 0.4, label="vLLM", color=VLLM, zorder=3)
    ax.bar([i + 0.2 for i in x], nano, 0.4, label="nano-vllm", color=NANO, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(10, max(vllm) * 6)
    for i, (n, v) in enumerate(zip(nano, vllm)):
        ax.text(i - 0.2, v * 1.12, f"{v:,.0f}", ha="center", fontsize=8, color="#475569")
        ax.text(i + 0.2, n * 1.12, f"{n:,.0f}", ha="center", fontsize=8, color=NANO)
        if n and v:
            ax.text(i, max(n, v) * 2.2, f"{v / n:.1f}\u00d7", ha="center",
                    fontsize=10, weight="bold", color="#475569")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("output tokens/s  (log)")
    ax.set_title(run_dir.name, fontsize=10)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", lw=0.5, color="#e2e8f0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2 — the arc. Percent of vLLM, so the ceiling is fixed at 100 and
    # progress is legible against the target rather than against itself.
    if rows:
        ax2 = axes[1]
        # One x position per run, in file order; a scenario missing from a run
        # leaves a gap rather than shifting the line left.
        runs_seen: list[tuple[str, str]] = []
        for r in rows:
            if (key := (r["date"], r["sha"])) not in runs_seen:
                runs_seen.append(key)
        for scenario in sorted({r["scenario"] for r in rows}):
            series = {(r["date"], r["sha"]): r["pct"] for r in rows if r["scenario"] == scenario}
            xs = [i for i, key in enumerate(runs_seen) if key in series]
            ax2.plot(xs, [series[runs_seen[i]] for i in xs], marker="o", label=scenario)
        ax2.axhline(100, ls="--", lw=1, color="#cbd5e1")
        ax2.text(len(runs_seen) - 0.55, 102, "vLLM", fontsize=8,
                 color="#94a3b8", ha="right")
        ax2.set_xticks(range(len(runs_seen)))
        ax2.set_xticklabels([f"{d[5:]}\n{s[:7]}" for d, s in runs_seen], fontsize=8)
        # One run is a scatter, not a line; pad so the dots are not on the spine.
        ax2.set_xlim(-0.5, len(runs_seen) - 0.5 if len(runs_seen) > 1 else 0.5)
        ax2.set_ylim(0, 112)
        ax2.set_ylabel("nano-vllm as % of vLLM")
        ax2.set_title("closing the gap (frozen spec)", fontsize=10)
        ax2.legend(frameon=False, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=3)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(args.out) if args.out else run_dir / "throughput.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
