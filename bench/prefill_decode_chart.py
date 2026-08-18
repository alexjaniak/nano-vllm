import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "experiments/2026-08-18-qwen-qwen3-8b-v1-5fa1f7e"
NANO, PRE, GREY = "#2563eb", "#fbbf24", "#94a3b8"
ORDER = ["fixed", "ragged", "ragged-sla", "prefix", "ragged-native"]

d = {}
for f in glob.glob(f"{RUN}/*__*.json"):
    eng, sc = os.path.basename(f)[:-5].split("__")
    d[(eng, sc)] = json.load(open(f))

names = [s for s in ORDER if ("nano-vllm", s) in d]
# prompt tok/s is not reported directly; derive it the same way the harness
# derives output_throughput — total tokens over the measured duration.
pre = [d[("nano-vllm", s)]["total_input_tokens"] / d[("nano-vllm", s)]["duration"] for s in names]
dec = [d[("nano-vllm", s)]["output_throughput"] for s in names]

fig, ax = plt.subplots(figsize=(9.5, 5))
x = range(len(names))
ax.bar([i - 0.21 for i in x], pre, 0.42, label="prefill  (prompt tokens)", color=PRE, zorder=3)
ax.bar([i + 0.21 for i in x], dec, 0.42, label="decode  (output tokens)", color=NANO, zorder=3)
for i, (p, o) in enumerate(zip(pre, dec)):
    ax.text(i - 0.21, p + 12, f"{p:,.0f}", ha="center", fontsize=9, color="#92400e")
    ax.text(i + 0.21, o + 12, f"{o:,.0f}", ha="center", fontsize=9, color=NANO, weight="bold")

avg = sum(dec) / len(dec)
ax.axhline(avg, ls="--", lw=1.2, color=NANO, alpha=0.55, zorder=2)
# Park the callout in the empty space above the short bars; on the right it
# lands on top of ragged-native.
ax.text(1.5, 165, f"decode never leaves ~{avg:.0f} tok/s",
        ha="center", fontsize=10.5, color=NANO, style="italic")
ax.annotate("", xy=(1.5, avg + 4), xytext=(1.5, 158),
            arrowprops=dict(arrowstyle="-", lw=1, color=NANO, alpha=0.55))

ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("tokens/s")
ax.set_ylim(0, max(pre) * 1.22)
ax.set_title("nano-vllm: prefill scales with the workload, decode does not\n"
             "Qwen3-8B · RTX 5090 · both engines capped at 8 concurrent",
             fontsize=11.5, loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.grid(axis="y", lw=0.5, color="#e2e8f0", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.985, 0.90, "vLLM decode, same runs: 605–640 tok/s",
        transform=ax.transAxes, ha="right", fontsize=10, color=GREY)
fig.tight_layout()
fig.savefig(f"{RUN}/prefill-vs-decode.png", dpi=150)
print(f"{RUN}/prefill-vs-decode.png")
