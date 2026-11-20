"""Plot the paired complete-answer and direction-specific comparisons in dark mode."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
audit = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 3.65), layout="constrained")
for ax in (left, right):
    ax.set_facecolor("#101721")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#435467", alpha=.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
fig.patch.set_facecolor("#101721")
keys = ("one_shot", "hold")
labels = ("One shot", "Seven-block hold")
colors = ("#72d9ed", "#f9bf84")
left.set_title("All four answers correct")
values = [100 * audit["counts"][arm] / audit["eligible"] for arm in keys]
bars = left.bar(labels, values, color=colors, width=.58)
left.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4)
right.set_title("Answers that change length")
right.set_xticks(np.arange(2), ["1 → 2 digits", "2 → 1 digits"])
for j, (arm, label, color) in enumerate(zip(keys, labels, colors, strict=True)):
    data = [100 * row["correct"][arm] / row["total"] for row in audit["directions"].values()]
    bars = right.bar(np.arange(2) + (j - .5) * .26, data, color=color, width=.23, label=label)
    right.bar_label(bars, labels=[f"{v:.1f}%" for v in data], padding=4)
right.legend(loc="lower right", frameon=False)
fig.savefig(root / "held-carry.png", dpi=160)
