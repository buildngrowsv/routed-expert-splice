"""Compare endpoint performance and the repairs gained and lost on the same panels."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent/"results"
s=json.loads((R/"summary.json").read_text())["test"]
fig,(a,b)=plt.subplots(1,2,figsize=(10.7,4.1),layout="constrained")
x=np.arange(3)
for name,values,c,marker in (("Adapted host",s["host"],"#304f75","o"),("Specialized donor",s["donor"],"#14776b","o"),
                             ("Donor change exchanged",s["shuffled_average_solves"],"#935a31","x")):
    a.plot(x,values,marker=marker,c=c,label=name,lw=1.5)
a.axhline(s["raw_donor"],color="#777",ls=":",label="Original donor")
a.set_ylim(75,155);a.set_ylabel("Correct repairs out of 384");a.set_title("Solves on unseen program families",loc="left",fontweight="bold",fontsize=12)
a.legend(fontsize=8,loc="center right",frameon=False)
for j,(name,k,c) in enumerate((("New solves","gains","#14776b"),("Previously correct, now wrong","losses","#935a31"))):
    bars=b.bar(x+(j-.5)*.36,s[k],width=.36,color=c,label=name)
    b.bar_label(bars,padding=2,fontsize=9)
b.set_ylim(0,129);b.set_title("What changed inside the donor?",loc="left",fontweight="bold",fontsize=12);b.set_ylabel("Repairs")
b.legend(fontsize=8,loc="upper right",frameon=False)
for ax in (a,b):
    ax.spines[["top","right"]].set_visible(False);ax.set_xticks(x,["101","211","307"]);ax.set_xlabel("Training order")
fig.savefig(R/"specialist.png",dpi=170,bbox_inches="tight")

