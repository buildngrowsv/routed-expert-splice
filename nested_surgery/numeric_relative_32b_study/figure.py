"""Plot complete carry groups under three matched activation policies."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
source = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.05, 3.6), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
arms = ("one_shot", "absolute", "relative", "relative_random")
names = ("One-shot edit", "Absolute hold", "Native-relative hold", "Orthogonal hold")
values = [100 * source["common_counts"][key] / source["common_native"] for key in arms]
bars = ax.bar(names, values, width=.63, color=("#72d9ed", "#edbb88", "#b8d898", "#abacba"))
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4)
ax.set_ylabel("Complete four-answer groups")
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_axisbelow(True)
ax.grid(axis="y", color="#44596b", alpha=.55)
ax.spines[["right", "top"]].set_visible(False)
fig.savefig(root / "moving-reference.png", dpi=160)
