"""Publication figure. All bars and dots come from verified item-level outcomes."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent/"results"
data=json.loads((R/"summary.json").read_text())
counts=data["counts"]
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={"width_ratios":[1.25,1]})
fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.79,wspace=.53)
fig.suptitle("Can the host acquire the donor’s Python repair skill?",x=.027,y=.98,ha="left",fontsize=18,weight="bold")
arms=["coder","general","shuffled","direct","knockout","native"]
names=["Coding donor","General donor","Shuffled donor","Direct affine","Removed","Native host"]
for i,a in enumerate(arms):
    before=counts["frozen_"+a][0]
    after=np.mean(counts["trained_"+a])
    ax.plot([before,after],[i,i],color="#b5c0c4",lw=2)
    ax.plot(before,i,"o",color="#b65e32",ms=7)
    ax.plot(after,i,"o",color="#126956",ms=7)
ax.set_yticks(range(6),names)
ax.invert_yaxis()
ax.set_xlim(0,400)
ax.set_xlabel("Test problems solved out of 384")
ax.set_title("A   Before and after training block 28",loc="left",fontsize=11,weight="bold",pad=12)
ax.plot([],[],"o",color="#b65e32",label="Frozen")
ax.plot([],[],"o",color="#126956",label="Task-trained")
ax.legend(loc="upper center",bbox_to_anchor=(.54,-.19),frameon=False,ncol=2)
bx.set_title("B   What happened to the new solves?",loc="left",fontsize=11,weight="bold",pad=12)
fields=["new","new_still_solved_off","new_lost_when_off"]
labs=["New vs. original host","Survived removal","Lost on removal"]
for i,(k,c) in enumerate(zip(fields,["#284d65","#126956","#b65e32"])):
    vals=[data["ledger_counts"][k][str(s)] for s in data["seed_order"]]
    m=np.mean(vals)
    bx.barh(i,m,color=c,height=.43)
    bx.text(m+4,i,str(round(m)) if len(set(vals))==1 else f"{m:.1f}",va="center",fontsize=11)
bx.set_yticks(range(3),labs)
bx.invert_yaxis()
bx.set_xlim(0,max(data["n"],20))
bx.set_xlabel("Mean number of problems across three seeds")
for axis in (ax,bx):
    axis.spines[["top","right","left"]].set_visible(False)
    axis.tick_params(axis="y",length=0)
    axis.grid(axis="x",alpha=.18)
    axis.set_axisbelow(True)
fig.savefig(R/"new-solves.png",dpi=170,facecolor="white")
