"""Plot complete carry groups and answer-length transitions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
background = "#101721"
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
fig.patch.set_facecolor(background)
labels = ("Borrowed", "Random", "Whole-state swap")
colors = ("#6ed3ed", "#d2a9a0", "#a8e8ad")
for ax in axes:
    ax.set_facecolor(background)
    ax.grid(axis="y", color="#344151", alpha=.55)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0, 113)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
names = ("both", "random", "whole")
complete = np.array([data["correct"][m] for m in names]) / data["eligible"]
axes[0].set_title("All four sums correct")
bars = axes[0].bar(labels, complete * 100, color=colors, width=.66)
axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in complete], padding=4)
axes[0].axhline(80, color="#ecbf6c", linewidth=1., linestyle="--")
axes[0].text(1.04, 81.6, "80% target", ha="center", color="#ecbf6c", fontsize=9)
axes[1].set_title("Completions that change length")
axes[1].set_xticks(range(2), ["1 to 2 digits", "2 to 1 digits"])
for j, mode in enumerate(names):
    directions = list(data["direction_counts"].values())
    rate = np.array([v["correct"][mode] / v["prompts"] for v in directions])
    bars = axes[1].bar(np.arange(2) + (j - 1) * .23, rate * 100, color=colors[j], width=.2)
    axes[1].bar_label(bars, labels=[f"{v:.1%}" for v in rate], padding=3, fontsize=8)
fig.savefig(root / "carry-32b.png", dpi=160)
