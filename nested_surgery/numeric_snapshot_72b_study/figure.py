"""Plot the read-context outcomes with the preservation threshold."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
run = json.loads((root / "audit.json").read_text())
n = run["eligible"]
data = run["paired_saved_and_current"]
order = ("selected", "unused", "orthogonal", "whole")
labels = ("Donor at relevant site · switch", "Donor at irrelevant site · preserve",
          "Orthogonal at relevant site · switch", "Full state at relevant site · switch")
colors = ("#84dfc2", "#e7bd85", "#a6abb0", "#b6acf0")
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10.5, 3.5), layout="constrained")
fig.patch.set_facecolor("#101916")
for j, (ax, title) in enumerate(zip(axes, ("Returns saved value", "Returns current value"), strict=True)):
    ax.set_facecolor("#101916")
    counts = [data[mode][j] for mode in order]
    y = np.arange(4)
    bars = ax.barh(y, counts, color=colors, height=.59)
    ax.bar_label(bars, labels=[f"{c}/{n}" for c in counts], padding=5, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 252)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Groups meeting the stated outcome")
    ax.set_title(title, loc="left", pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.24)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
axes[0].invert_yaxis()
fig.savefig(root / "snapshot.png", dpi=170)
