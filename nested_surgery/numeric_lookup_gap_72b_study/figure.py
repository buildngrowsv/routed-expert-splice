"""Plot the exact-distance control with both archived comparisons."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/"audit.json").read_text())
n=r["shared"]
plt.style.use("dark_background")
f,ax=plt.subplots(figsize=(8.8,3.45),layout="constrained")
f.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
x=np.arange(3)
for j,(m,color,label) in enumerate((("donor","#84dfc2","Seven-coordinate donor"),
                                      ("whole","#b6acf0","Whole state"),
                                      ("orthogonal","#d5b48c","Orthogonal"))):
    y=[r["switches"][f"{k}_{m}"] for k in ("early","late","gap")]
    bars=ax.bar(x+(j-1)*.24,y,width=.20,color=color,label=label)
    ax.bar_label(bars,labels=[f"{v}/{n}" for v in y],padding=3,fontsize=8)
ax.set_xticks(x,["Early key","Late key","Late key with matched gap"])
ax.set_ylim(0,1.14*n)
ax.set_ylabel("Both mirrored contexts switch")
ax.set_axisbelow(True)
ax.grid(axis="y",alpha=.24)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=8)
f.savefig(root/"key-gap.png",dpi=170)
