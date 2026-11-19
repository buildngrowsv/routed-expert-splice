import json
from pathlib import Path
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/"results"
a=json.loads((R/"test.json").read_text()); b=json.loads((R/"partial-test.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(10.4,4.4))
fig.subplots_adjust(left=.14,right=.98,top=.78,bottom=.18,wspace=.68)
fig.suptitle("Can a router choose the right coding model?",x=.03,y=.97,ha="left",weight="bold",fontsize=18)
one=[("Host",a["host"]["solves"]),("Donor",a["donor"]["solves"]),("Confidence route",a["routed"]["solves"]),("Oracle",a["oracle"])]
two=[("Donor",b["solves"]["donor"]),("First valid choice",b["solves"]["first_consistent"]),("Checked donor",b["solves"]["donor_checked"]),("Checked route",b["solves"]["cascade"])]
for ax,entries,title in zip(axes,(one,two),("A   Confidence as a routing signal","B   One ambiguous example as a check")):
    for i,(name,val) in enumerate(entries):
        ax.barh(i,val,color=("#8c999b","#8c999b","#187666","#31546b")[i],height=.57)
        ax.text(val+4,i,str(val),va="center")
    ax.set(yticks=range(4),yticklabels=[n for n,v in entries],xlim=(0,300),xlabel="Problems solved out of 512")
    ax.invert_yaxis()
    ax.set_title(title,loc="left",fontsize=10,weight="bold")
    ax.spines[["right","top","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
fig.savefig(R/"routing.png",dpi=160,facecolor="white")
