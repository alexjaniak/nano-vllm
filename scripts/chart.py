#!/usr/bin/env python3
"""Plot output throughput against arrival rate for both engines.

The one picture the sweep is for: vLLM's curve climbs with offered load until
it saturates, nano-vllm's is a flat line. Reads the same result JSONs as
report.py.

    python3 scripts/chart.py [experiments/<name>/sweep] [-o out.png]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FILENAME = re.compile(r"^(?P<engine>.+)_rate(?P<rate>[^_]+)\.json$")

# Validated categorical slots 1 and 2 (light mode) — see the dataviz palette.
# node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> all pass.
SERIES = {"vllm": "#2a78d6", "nano-vllm": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e3e0"


def newest_run() -> Path | None:
    """Newest run directory: a curated experiments/<name>/<phase> first, else
    a raw results/<stamp> as sweep.sh writes it on the box."""
    candidates = sorted(Path("experiments").glob("*/*/")) + sorted(Path("results").glob("*/"))
    runs = [d for d in candidates if any(d.glob("*_rate*.json"))]
    return runs[-1] if runs else None


def rate_key(rate: str) -> float:
    return float("inf") if rate == "inf" else float(rate)


def load(run_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in sorted(run_dir.glob("*.json")):
        if not (m := FILENAME.match(path.name)):
            continue
        blob = json.loads(path.read_text())
        out.setdefault(m["engine"], {})[m["rate"]] = blob["output_throughput"]
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out_path = Path("throughput.png")
    if "-o" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("-o") + 1])
    run_dir = Path(args[0]) if args else newest_run()

    data = load(run_dir)
    if not data:
        print(f"no result JSONs in {run_dir}", file=sys.stderr)
        return 1
    rates = sorted({r for by_rate in data.values() for r in by_rate}, key=rate_key)
    x = list(range(len(rates)))

    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive horizontal grid only; the x positions are ordinal categories.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=10)

    for engine, color in SERIES.items():
        if engine not in data:
            continue
        y = [data[engine].get(r, float("nan")) for r in rates]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # Direct label at the line end: a colored mark carries identity, the
        # text itself stays in ink (never the series color).
        ax.annotate(engine, xy=(x[-1], y[-1]), xytext=(12, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=11, color=INK, fontweight="medium")
        ax.annotate(f"{y[-1]:.0f}", xy=(x[-1], y[-1]), xytext=(12, -14),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=10, color=INK_MUTED)

    # The headline, stated once rather than as a number on every point.
    if {"vllm", "nano-vllm"} <= data.keys():
        top, bottom = data["vllm"][rates[-1]], data["nano-vllm"][rates[-1]]
        ax.annotate(
            "", xy=(len(x) - 1.35, top), xytext=(len(x) - 1.35, bottom),
            arrowprops=dict(arrowstyle="<->", color=INK_MUTED, linewidth=1.2))
        ax.annotate(f"{top / bottom:.1f}×", xy=(len(x) - 1.32, (top + bottom) / 2),
                    fontsize=15, color=INK, fontweight="bold", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(["∞" if r == "inf" else r for r in rates])
    ax.set_xlim(-0.25, len(x) - 0.28)
    ax.set_ylim(0, max(v for d in data.values() for v in d.values()) * 1.14)
    ax.set_xlabel("offered load (requests/sec)", fontsize=10.5, color=INK_MUTED,
                  labelpad=9)
    ax.set_ylabel("output tokens/sec", fontsize=10.5, color=INK_MUTED, labelpad=9)
    # Title block drawn as figure text so the subtitle sits under the title
    # rather than above it (ax.set_title would own the top of the axes).
    fig.text(0.105, 0.935, "Throughput vs offered load — Qwen3-8B, 1×RTX 5090",
             fontsize=14.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.105, 0.862,
             "Both engines capped at max_num_seqs=8. Generation tokens only.",
             fontsize=10, color=INK_MUTED, ha="left", va="top")

    fig.subplots_adjust(left=0.105, right=0.845, top=0.80, bottom=0.145)
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
