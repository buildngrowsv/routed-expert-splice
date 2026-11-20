"""Plot switches by edit and placement."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
record = json.loads((root/"audit.json").read_text())
n = record["eligible"]
keys = ("early_donor","late_donor","late_orthogonal","late_whole")
labels = ("Early key, donor","Late key, donor","Late key, orthogonal","Late key, whole state")
counts = [record["mirrored_switches"][m] for m in keys]
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(7.8,3.3),layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
bars=ax.barh(labels,counts,color=("#a9b6ad","#84dfc2","#a9b6ad","#b6acf0"),height=.57)
ax.bar_label(bars,labels=[f"{v}/{n}" for v in counts],padding=5,fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0,161)
ax.set_xticks([0,35,70,105,139])
ax.set_xlabel("Shared native groups that switched both mirrored lookups")
ax.set_axisbelow(True)
ax.grid(axis="x",alpha=.21)
ax.spines[["left","top","right"]].set_visible(False)
ax.tick_params(axis="y",length=0,pad=7)
fig.savefig(root/"late-donor.png",dpi=170)
