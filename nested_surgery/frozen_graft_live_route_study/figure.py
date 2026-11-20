"""Plot synchronized batch latency for all three execution paths."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
data = json.loads((here / "audit.json").read_text())
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8,4.2),facecolor="#101916",layout="constrained")
ax.set_facecolor("#101916")
x=np.arange(2)
for i,(name,color) in enumerate((("Standalone 7B","#b6acf0"),("Always blend","#e8bc79"),("Confidence route","#84dfc2"))):
    bars=ax.bar(x+(i-1)*.23,[data[k]["median_ms"][i] for k in ("neutral","decoys")],
                width=.21,label=name,color=color)
    ax.bar_label(bars,fmt="%.1f",padding=3)
ax.set_xticks(x,["Neutral controls","Bracket decoys"])
ax.set(title="Selective grafting cut measured latency versus always blending",
       ylabel="Median synchronized batch latency (ms)",ylim=(0,540))
ax.legend(frameon=False,ncol=3,loc="upper center")
ax.grid(axis="y",alpha=.18)
fig.savefig(here/"live-timing.png",dpi=170)
