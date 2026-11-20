"""Plot mean margins for the five reference conditions in both models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
audit = json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.9,4.4), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
conditions = ("native","learned_clean","learned_swap","random_clean","random_swap")
labels = ("Untouched", "Learned, original", "Learned, partner", "Random, original", "Random, matched edit")
for offset, key, label, color in ((-.15,"donor","Qwen coder 7B","#85dabc"),(.15,"host","Qwen general 14B","#a7b8d7")):
    vals = [audit[key]["mean_margin"][arm] for arm in conditions]
    bars = ax.barh([i+offset for i in range(5)], vals, height=.27, label=label, color=color)
    ax.bar_label(bars, fmt="%+.1f", padding=4)
ax.axvline(0, color="#d2dae3", lw=1)
ax.set(yticks=range(5), yticklabels=labels, xlim=(-16,16),
       xlabel="Mean logit margin for the original letter")
ax.legend(loc="upper center", bbox_to_anchor=(.5,-.21), ncol=2, frameon=False)
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"tunnel.png", dpi=165)
