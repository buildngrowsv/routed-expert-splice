"""Compare overwrite-only switches, preservation and wrong-value intrusions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())
n = data["eligible"]
fields = ("current_switches", "saved_preserved", "overwritten_answer")
labels = ("Current value\nswitches", "Saved value\nis preserved", "Edited overwrite intrudes\non a saved-value answer")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9.2, 4.15), layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
arms = (("donor", "Seven-dimensional donor", "#84dfc2"),
        ("whole", "Matched full-state exchange", "#b6acf0"),
        ("orthogonal", "Equal-norm orthogonal", "#a5aea8"))
for i, (mode, label, color) in enumerate(arms):
    counts = [data[key][mode] for key in fields]
    bars = ax.bar(np.arange(3) + (i-1)*.25, counts, width=.21, color=color, label=label)
    ax.bar_label(bars, labels=[str(k) for k in counts], padding=4, fontsize=8)
ax.set_xticks(range(3), labels)
ax.set_ylim(0, 1.16*n)
ax.set_ylabel(f"Groups out of {n} solved natively")
ax.grid(axis="y", alpha=.24)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, frameon=False, fontsize=8)
fig.savefig(root / "overwrite.png", dpi=170)
