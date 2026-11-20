"""Plot exact triplet solves by prompt format."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
scores = json.loads((here/"results/result.json").read_text())["complete"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.3,4.1), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for offset, key, label, color in ((-.18,"donor","Qwen coder 7B","#85dabc"),(.18,"host","Qwen general 14B","#a7b8d7")):
    heights = [scores[f"{key}_{arm}"] for arm in ("fim","empty","prefix")]
    bars = ax.bar([i+offset for i in range(3)], heights, width=.32, label=label, color=color)
    ax.bar_label(bars, padding=4)
ax.set(xticks=range(3), xticklabels=["Markers + suffix","Markers, empty suffix","Plain Python"],
       ylim=(0,256), ylabel="Groups with all three repairs correct")
ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"format.png", dpi=165)
