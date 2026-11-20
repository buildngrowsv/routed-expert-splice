"""Plot side-by-side detection and reversal in the three model scales."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8.8,3.7),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
names = ("285M custom","Qwen coder 7B","Qwen general 14B")
full = [458/461*100,100*d["donor"]["switches"]["whole_site"]/512,100*d["host"]["switches"]["whole_site"]/512]
for offset,values,label,color in ((-.23,[100,100,100],"Decode from a slice","#a8b7d4"),
                                 (0,[0,0,0],"Reverse with one slice","#eec18c"),
                                 (.23,full,"Reverse with all channels","#83d8b8")):
    y = [i+offset for i in range(3)]
    bar = ax.barh(y,values,height=.21,label=label,color=color)
    ax.bar_label(bar,labels=[f"{p:.1f}%" if 0<p<100 else f"{p:.0f}%" for p in values],padding=5)
ax.set(yticks=range(3),yticklabels=names,xlim=(0,112),xticks=range(0,101,20),
       xlabel="Both letters decoded correctly or both answers reversed (%)")
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.21),ncol=3,frameon=False,fontsize=9)
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"identifier-widths.png",dpi=165)
