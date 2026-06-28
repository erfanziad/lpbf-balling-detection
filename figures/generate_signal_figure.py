"""
Generate Cohen's d per layer figure.
V=1850 mm/s for L226-L231 (corrected from 1800).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cohen's d values per layer (from enriched_df analysis)
layers = [226, 227, 228, 229, 230, 231, 245, 246, 247, 248, 249, 250, 251, 252]
cohens_d = [-0.191, +0.646, +0.213, -0.002, +0.222, +0.083,
            -0.139, -0.224, -0.084, +0.109, -0.155, -0.282, +0.216, -0.290]
powers   = [360, 380, 400, 420, 440, 460,
            320, 340, 360, 380, 400, 420, 440, 460]
speeds   = [1850]*6 + [2000]*8

# Colors: red for V=1850, blue for V=2000
colors = ["#C62828" if v == 1850 else "#1565C0" for v in speeds]

fig, ax = plt.subplots(figsize=(13, 5))

x = np.arange(len(layers))
bars = ax.bar(x, cohens_d, color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.8, width=0.65)

# Zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
# Detectability threshold reference
ax.axhline(0.2, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
ax.axhline(-0.2, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

# X axis labels: "L226\n360W\n1850"
xlabels = [f"L{l}\n{p}W\nV={v}" for l, p, v in zip(layers, powers, speeds)]
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8)

ax.set_ylabel("Cohen's $d$", fontsize=11)
ax.set_xlabel("Layer", fontsize=11)
ax.set_title(
    "Raw camera signal strength per layer\n"
    "Cohen's $d$ = normalized difference in frame-to-frame pixel activity "
    "(balling vs clean frames)\n"
    "Positive = balling frames more visually active than clean frames",
    fontsize=10
)

# Value labels on bars
for bar, d in zip(bars, cohens_d):
    ypos = d + 0.02 if d >= 0 else d - 0.05
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
            fontsize=7.5, color="black")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#C62828", alpha=0.85, label="V = 1850 mm/s"),
    Patch(facecolor="#1565C0", alpha=0.85, label="V = 2000 mm/s"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

ax.set_ylim(-0.5, 0.85)
ax.grid(axis="y", alpha=0.3, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("a2_signal_cohens_d.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: a2_signal_cohens_d.png")
