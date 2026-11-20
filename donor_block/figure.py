"""Render final repair scores and the within-donor expert-reversion intervention."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent/"results"
s=json.loads((R/"confirmation-summary.json").read_text())
moves=json.loads((R/"confirmation-switches.json").read_text())
seeds=("101","211","307")
keys=("donor_base","previous_expert","attention","expert","full","host","shortcut")
names=("Frozen donor","Expert trained alone","Adapted attention + norms","Jointly trained expert alone",
       "Whole donor block","Adapted larger host","Output-count lookup")
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.5),layout="constrained",gridspec_kw={"width_ratios":[1.35,1]})
for y,(k,label) in enumerate(zip(keys,names)):
    values=[s["scores"][k]["solved"]] if k in s["scores"] else [s["scores"][f"{k}_{j}"]["solved"] for j in seeds]
    c="#306276" if k=="host" else "#ab5e32" if k=="shortcut" else "#287a6f"
    a.barh(y,np.mean(values),height=.54,color=c,alpha=.84)
    if len(values)>1:a.plot([min(values),max(values)],[y,y],color="#17252b",lw=2,solid_capstyle="round")
    a.text(max(values)+8,y,str(values[0]) if len(set(values))==1 else f"{min(values)}–{max(values)}",va="center",fontsize=8)
a.invert_yaxis();a.set_yticks(range(len(keys)),names);a.set_xlim(0,512);a.set_xlabel("Correct repairs out of 512")
a.set_title("Confirmation on new control-flow families",loc="left",fontsize=11,fontweight="bold")
x=np.arange(3); d=[m["attention"] for m in [moves[j] for j in seeds]]
for side,key,color,title in ((-.19,"full_only","#a8582c","Correct repairs lost"),(.19,"other_only","#2a7b70","Incorrect repairs rescued")):
    v=[len(r[key]) for r in d]
    bars=b.bar(x+side,v,.36,color=color,label=title)
    b.bar_label(bars,padding=2,fontsize=9)
b.set_xticks(x,seeds);b.set_xlabel("Training order");b.set_ylabel("Number of repairs");b.set_ylim(0,157)
b.set_title("Restoring the original feed-forward weights",loc="left",fontsize=11,fontweight="bold")
b.legend(frameon=False,fontsize=8,loc="upper right")
for ax in (a,b):ax.spines[["top","right"]].set_visible(False)
fig.savefig(R/"component-restoration.png",dpi=170,bbox_inches="tight")

