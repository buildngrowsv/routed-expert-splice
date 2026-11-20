"""Plot original-key blocking, the syntax control, and unrestricted attention."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
run=json.loads((root/"audit.json").read_text())
arms=("unrestricted","syntax_block","key_block")
names=("Unrestricted","Mask syntax reads","Mask key reads")
colors=("#b6acf0","#a4ada7","#84dfc2")
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(9.3,3.54),layout="constrained")
fig.patch.set_facecolor("#101916")
items=((run["native"],256,"Native solves, all groups"),
       (run["switched"],run["eligible"],"Both mirrored lookups switch"))
for ax,(scores,n,title) in zip(axes,items,strict=True):
    ax.set_facecolor("#101916")
    v=[scores[k] for k in arms]
    bars=ax.bar(range(3),[100*x/max(n,1) for x in v],width=.57,color=colors)
    ax.bar_label(bars,labels=[f"{x}/{n}" for x in v],padding=4,fontsize=8)
    ax.set_ylim(0,112)
    ax.set_yticks([0,25,50,75,100],["0%","25%","50%","75%","100%"])
    ax.set_xticks(range(3),names,fontsize=8)
    ax.set_title(title,loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="y",alpha=.24)
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"dictionary-mask.png",dpi=170)
