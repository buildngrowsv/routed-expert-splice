"""Plot exact group outcomes by intervention."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
results = json.loads((root / "audit.json").read_text())
n = results["eligible"]
names = ["selected", "unused", "orthogonal", "whole"]
labels = ["Donor at read assignment\nSwitch answer", "Donor at unused assignment\nPreserve answer",
          "Orthogonal at read assignment\nSwitch answer", "Full state at read assignment\nSwitch answer"]
counts = [results["counts"][name] for name in names]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.9, 3.8), layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
bars = ax.barh(labels, counts, color=("#84dfc2", "#e7bd85", "#a6abb0", "#b6acf0"), height=.58)
ax.bar_label(bars, labels=[f"{count}/{n}" for count in counts], padding=6, fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 295)
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_xlabel("Groups satisfying the stated outcome in both read contexts")
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=.22)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=12)
fig.savefig(root / "selection.png", dpi=170)
