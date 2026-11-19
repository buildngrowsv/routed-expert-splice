"""Plot the forced replacement panel on its shared, held-out contexts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/"results"/"forced"
s=json.loads((p/"summary.json").read_text())["results"]
s["direct"]=json.loads((p/"direct.json").read_text())["score"]
arms=["native","intact","direct","shuffled","knockout"]
names=["Original host","Intact donor","Direct linear map","Shuffled donor","Block removed"]
colors=["#243b53","#147d67","#6d7890","#917093","#ac6756"]
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained",sharey=True)
for ax,key,label,lo,hi in [(axes[0],"code","Python source",1.48,2.07),(axes[1],"docs","Documentation",1.88,2.55)]:
    for i,(arm,c) in enumerate(zip(arms,colors)):
        x=s[arm][key]["nll"];y=4-i
        ax.plot(x,y,"o",ms=9,color=c);ax.text(x+.013,y,f"{x:.3f}",va="center",fontsize=10)
    ax.set_xlim(lo,hi);ax.set_ylim(-.6,4.6);ax.set_yticks(range(5),names[::-1]);ax.grid(axis="x",alpha=.15)
    ax.set_title(label,loc="left",weight="bold");ax.set_xlabel("Next-token loss (nats, lower is better)")
    ax.spines[["top","right","left"]].set_visible(False)
fig.suptitle("An intact donor recovered part of the removed block’s function",fontsize=15,weight="bold")
fig.savefig(p/"expert-replacement.png",dpi=165)
