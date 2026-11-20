"""Show the complete digit-path result in the two recipients."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.1), layout="constrained")
fig.patch.set_facecolor("#101721")
for ax in (left, right):
    ax.set_facecolor("#101721")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 113)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.45, color="#4a5b72")

left.set_title("Complete quartets on shared prompts")
left.set_xticks(range(2), ["32B", "72B"])
right.set_title("72B answers that change length")
right.set_xticks(range(2), ["One to two digits", "Two to one digit"])
for j, (arm, label, color) in enumerate(zip(("both", "random", "whole"),
        ("Donor edit", "Orthogonal", "Full-state swap"), ("#70d8ef", "#b3b4c7", "#fac08b"), strict=True)):
    groups = [data["shared_counts"][size][arm] for size in ("32b", "72b")]
    bars = left.bar(np.arange(2) + (j - 1)*.23, 100*np.array(groups)/data["shared_native"], width=.21,
                    label=label, color=color)
    left.bar_label(bars, labels=groups, padding=4, fontsize=8)
    endpoints = [v["correct"][arm] for v in data["direction_counts"].values()]
    totals = [v["prompts"] for v in data["direction_counts"].values()]
    bars = right.bar(np.arange(2) + (j - 1)*.23, 100*np.array(endpoints)/totals, width=.21, color=color)
    right.bar_label(bars, labels=endpoints, padding=4, fontsize=8)
left.legend(loc="upper center", bbox_to_anchor=(.5,-.15), ncol=3, frameon=False, fontsize=9)
fig.savefig(root / "carry-72b.png", dpi=160)
