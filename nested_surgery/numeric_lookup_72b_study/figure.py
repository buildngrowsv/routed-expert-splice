"""Plot dictionary routing outcomes and the registered target."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())
n = data["eligible"]
scores = data["counts"]
modes = ("donor", "orthogonal", "whole")
names = ("Seven-dimensional donor", "Equal-norm orthogonal", "Full-state exchange")
colors = ("#84dfc2", "#929e98", "#b6acf0")
labels = ("Both mirrored contexts switch", "Constant table preserves")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.7, 4.1), layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
for j, (mode, name, color) in enumerate(zip(modes, names, colors, strict=True)):
    values = [scores[mode][k] for k in ("both_switch", "inert")]
    boxes = ax.bar(np.arange(2) + (j - 1)*.235, values, width=.202, label=name, color=color)
    ax.bar_label(boxes, labels=[f"{v}/{n}" for v in values], padding=4, fontsize=9)
ax.plot((-.37, .37), (.8*n, .8*n), color="#e7bd85", linestyle="--", linewidth=1.4)
ax.annotate("80% switching threshold", (-.35, .8*n), xytext=(0, 6), textcoords="offset points",
            fontsize=8, color="#e7bd85")
ax.set_xticks(range(2), labels)
ax.set_ylim(0, 218)
ax.set_ylabel("Groups meeting the stated outcome")
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.24)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.115), ncol=3, frameon=False, fontsize=9)
fig.savefig(root / "lookup.png", dpi=170)
