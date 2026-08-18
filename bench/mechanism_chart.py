import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "experiments/2026-08-18-qwen-qwen3-8b-v1-5fa1f7e"
NANO, VLLM, WARM = "#2563eb", "#94a3b8", "#f59e0b"

# Solved from the two measured ITLs at batch 8:
#   v0   8*(dense+attn) = 171.9   ->  dense+attn = 21.5
#   now  dense + 8*attn = 118.0   ->  attn = 13.8, dense = 7.7
ATTN, DENSE = 13.8, 7.7
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))

bars = ["v0\n8 separate\nforward passes", "now\n1 packed\nforward pass"]
dense = [8 * DENSE, DENSE]
attn = [8 * ATTN, 8 * ATTN]
a.bar(bars, dense, 0.5, label="dense layers (shared)", color=NANO, zorder=3)
a.bar(bars, attn, 0.5, bottom=dense, label="attention (still serial)", color=WARM, zorder=3)
for i, (d, t) in enumerate(zip(dense, attn)):
    a.text(i, d / 2, f"{d:.0f} ms", ha="center", va="center", color="white", fontsize=10, weight="bold")
    a.text(i, d + t / 2, f"{t:.0f} ms", ha="center", va="center", color="white", fontsize=10, weight="bold")
    a.text(i, d + t + 4, f"{d + t:.0f} ms total", ha="center", fontsize=9, color="#475569")
a.set_ylabel("time per decode step, batch of 8 (ms)")
a.set_title("Where the step time goes  (derived from measured ITL)", fontsize=10)
a.set_ylim(0, 200)
a.legend(frameon=False, fontsize=9)
a.grid(axis="y", lw=0.5, color="#e2e8f0", zorder=0)
a.spines[["top", "right"]].set_visible(False)

gpu = {(r["engine"], r["scenario"]): r for r in csv.DictReader(open(f"{RUN}/gpu.csv"))}
metrics = [("dram_active", "memory interface\nactive"), ("tensor_active", "tensor pipe\nactive")]
xs = range(len(metrics))
v = [float(gpu[("vllm", "ragged")][m]) for m, _ in metrics]
n = [float(gpu[("nano-vllm", "ragged")][m]) for m, _ in metrics]
b.bar([i - 0.2 for i in xs], v, 0.4, label="vLLM", color=VLLM, zorder=3)
b.bar([i + 0.2 for i in xs], n, 0.4, label="nano-vllm", color=NANO, zorder=3)
for i, (vv, nn) in enumerate(zip(v, n)):
    b.text(i - 0.2, vv + 0.012, f"{vv:.3f}", ha="center", fontsize=9, color="#475569")
    b.text(i + 0.2, nn + 0.012, f"{nn:.3f}", ha="center", fontsize=9, color=NANO)
b.set_xticks(list(xs)); b.set_xticklabels([lbl for _, lbl in metrics])
b.set_ylabel("ratio of cycles active")
b.set_ylim(0, 0.48)
b.set_title("The GPU is idle on both axes  (ragged scenario)", fontsize=10)
b.legend(frameon=False, fontsize=9)
b.grid(axis="y", lw=0.5, color="#e2e8f0", zorder=0)
b.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(f"{RUN}/mechanism.png", dpi=150)
print(f"{RUN}/mechanism.png")
