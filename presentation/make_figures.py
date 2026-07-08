"""Presentation figures: scaling plot + latency/quality pareto.

Data from eval/results/registry.jsonl + data/reports/ (see those for eval-set
caveats; the 4B pre-RL point is on kg200, others on pilot50/val366).
Run: .venv/bin/python presentation/make_figures.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"
GREY = "#8a8a8a"
GRID = "#e3e3e3"
MUT = "#6b6b6b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUT,
    "ytick.color": MUT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---------------------------------------------------------------- figure 1
# params (B) -> NDCG. pre-RL in grey, post-RL in black.
pre = {0.6: 0.060, 1.7: 0.129, 4.0: 0.188}   # 0.6B/4B pilot-style sets, 1.7B = val@step0
post = {1.7: 0.259}                            # val@step40, still training
anecdote = (9.0, 0.86)                         # Qwen3.5-9B, n=1 -- shown, not fitted

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

x = np.array(sorted(pre))
y = np.array([pre[k] for k in x])
# fit in log-param space
coef = np.polyfit(np.log10(x), y, 1)
xs = np.logspace(np.log10(0.5), np.log10(10), 100)
ax.plot(xs, coef[0] * np.log10(xs) + coef[1], color=GREY, lw=2, zorder=2,
        label="pre-RL (untrained + tools)")
ax.scatter(x, y, s=70, color=GREY, zorder=3)

# post-RL: single point; black line reuses the pre-RL slope through it,
# solid where we have evidence-adjacent range, to visualize the lift.
px, py = 1.7, post[1.7]
b_post = py - coef[0] * np.log10(px)
ax.plot(xs, coef[0] * np.log10(xs) + b_post, color=INK, lw=2.4, zorder=4,
        label="post-RL (60 GRPO steps; slope assumed)")
ax.scatter([px], [py], s=110, color=INK, zorder=5)

# the lift arrow at 1.7B
ax.annotate("", xy=(px, py - 0.008), xytext=(px, pre[1.7] + 0.008),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.annotate("+0.13 NDCG\n(~90 min RL, 1x A100)", xy=(px * 0.88, (py + pre[1.7]) / 2),
            fontsize=9.5, color=INK, va="center", ha="right")

# 9B anecdote: hollow marker, clearly flagged
ax.scatter([anecdote[0]], [anecdote[1]], s=80, facecolors="white",
           edgecolors=GREY, lw=1.6, zorder=3)
ax.annotate("Qwen3.5-9B (n=1,\nnot fitted)", xy=(anecdote[0], anecdote[1]),
            xytext=(anecdote[0] * 0.52, anecdote[1] - 0.03), fontsize=9, color=MUT)

# frontier reference band
for name, v, dy in (("MiniMax-M3  0.456", 0.456, 0.012), ("DeepSeek-v4-pro  0.387", 0.387, -0.030)):
    ax.axhline(v, color=GRID, lw=1.2, zorder=1)
    ax.annotate(name, xy=(0.52, v + dy), fontsize=9, color=MUT)

for k in pre:
    ax.annotate(f"{pre[k]:.3f}", xy=(k, pre[k] - 0.034), fontsize=9, color=MUT, ha="center")
ax.annotate(f"{py:.3f}", xy=(px, py + 0.014), fontsize=10, color=INK, ha="center", fontweight="bold")

ax.set_xscale("log")
ax.set_xticks([0.6, 1.7, 4, 9])
ax.set_xticklabels(["0.6B", "1.7B", "4B", "9B"])
ax.set_xlim(0.45, 11)
ax.set_ylim(0, 0.95)
ax.set_xlabel("policy parameters (log scale)")
ax.set_ylabel("two-tier NDCG@50 (held-out)")
ax.set_title("RL lift vs base-model scale — Qwen3 + graph tools", fontsize=12, pad=12)
ax.grid(axis="y", color=GRID, lw=0.7)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("presentation/fig_scaling.png", bbox_inches="tight")

# ---------------------------------------------------------------- figure 2
# latency (s/question) vs NDCG pareto. Shapes: circle=local policy, square=API.
pts = [
    # name, s/q, ndcg, kind(local/api/floor), trained
    ("vector-only", 0.6, 0.073, "floor", False),
    ("Qwen3-0.6B", 6.1, 0.060, "local", False),
    ("Qwen3-4B", 14.9, 0.188, "local", False),
    ("Qwen3.5-9B (n=1)", 31.1, 0.86, "local", False),
    ("MiniMax-M3", 21.0, 0.456, "api", False),
    ("DeepSeek-v4-pro", 47.1, 0.387, "api", False),
    ("Qwen3-1.7B +RL", 8.0, 0.259, "local", True),  # latency provisional: post-run eval pending
]

fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=200)
for name, s, nd, kind, trained in pts:
    color = INK if trained else GREY
    marker = "s" if kind == "api" else ("D" if kind == "floor" else "o")
    face = color if (trained or kind != "local" or True) else "white"
    ax2.scatter([s], [nd], s=110 if trained else 80, marker=marker,
                color=color, zorder=4 if trained else 3)
    dx, dy, ha, va = 1.13, 0, "left", "center"
    if name.startswith("DeepSeek"):
        dx, ha = 0.88, "right"
    if name == "vector-only":
        dx, dy, ha, va = 1.0, 0.035, "center", "bottom"
    ax2.annotate(name, xy=(s * dx, nd + dy), fontsize=9.5,
                 color=INK if trained else MUT, va=va, ha=ha,
                 fontweight="bold" if trained else "normal")

# pareto frontier (non-dominated: lower latency, higher ndcg)
frontier = sorted([(s, nd) for _, s, nd, _, _ in pts], key=lambda t: t[0])
best = -1
fx, fy = [], []
for s, nd in frontier:
    if nd > best:
        fx.append(s); fy.append(nd); best = nd
ax2.plot(fx, fy, color=GRID, lw=1.4, ls="--", zorder=1, drawstyle="steps-post")

ax2.set_xscale("log")
ax2.set_xticks([1, 3, 10, 30])
ax2.set_xticklabels(["1s", "3s", "10s", "30s"])
ax2.set_xlim(0.4, 90)
ax2.set_ylim(0, 0.95)
ax2.set_xlabel("end-to-end latency per question (log scale)")
ax2.set_ylabel("two-tier NDCG@50")
ax2.set_title("Latency vs quality — the axis we actually care about", fontsize=12, pad=12)
ax2.grid(axis="y", color=GRID, lw=0.7)
ax2.spines[["top", "right"]].set_visible(False)
ax2.annotate("serving stacks differ (transformers/vllm/API);\n1.7B+RL latency provisional until post-run eval",
             xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8, color=MUT, va="top")
fig2.tight_layout()
fig2.savefig("presentation/fig_pareto.png", bbox_inches="tight")
print("wrote presentation/fig_scaling.png, presentation/fig_pareto.png")
