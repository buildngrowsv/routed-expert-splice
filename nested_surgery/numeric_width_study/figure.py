"""Plot effect of the same seven-dimensional alignment in both model widths."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit-paired.json").read_text())
n = data["shared_native"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.75, 4.2), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
groups = ("stitch", "random", "whole")
labels = ("Matched-input alignment", "Equal-norm orthogonal", "Whole-state swap")
for recipient, offset, color, legend in (
        ("32b", -.16, "#72d9ed", "32B · 5,120 values"), ("72b", .16, "#ffc182", "72B · 8,192 values")):
    counts = [data["counts"][recipient][name] for name in groups]
    bars = ax.bar(np.arange(3) + offset, [100*c/n for c in counts], width=.29, label=legend, color=color)
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=4)
ax.set_ylim(0, 117)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xticks(np.arange(3), labels)
ax.set_ylabel(f"Both answers reversed on {n} shared native pairs")
ax.grid(axis="y", color="#39526b", alpha=.45)
ax.spines[["right", "top"]].set_visible(False)
ax.set_axisbelow(True)
ax.legend(loc="upper center", frameon=False, ncol=2)
fig.savefig(root / "cross-width.png", dpi=160)
