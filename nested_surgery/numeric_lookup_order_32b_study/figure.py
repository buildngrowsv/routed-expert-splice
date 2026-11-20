"""Plot switches in the 32B cohort and in the matched two-host cohort."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/"audit.json").read_text())
keys=("early","late","gap")
titles=("Early key","Late key","Late, matched gap")
plt.style.use("dark_background")
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.55),layout="constrained")
fig.patch.set_facecolor("#101916")
for ax in (a,b):
    ax.set_facecolor("#101916")
    ax.set_ylim(0,113)
    ax.set_yticks([0,25,50,75,100],["0%","25%","50%","75%","100%"])
    ax.set_xticks(range(3),titles,fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y",alpha=.25)
n=r["triple"]["n"]
v=[r["triple"]["switches"][k] for k in keys]
bars=a.bar(range(3),100*np.array(v)/n,color=("#a6afa8","#84dfc2","#d9ba89"),width=.6)
a.bar_label(bars,labels=[f"{x}/{n}" for x in v],padding=4,fontsize=8)
a.set_title("32B shared across three layouts",loc="left")
a.set_ylabel("Both mirrored contexts switch")
n=r["posthoc_all_six_native_groups"]
for j,(model,color) in enumerate((("32b","#84dfc2"),("72b","#b6acf0"))):
    v=[r["posthoc_shared_switches"][model][k] for k in keys]
    bars=b.bar(np.arange(3)+(j-.5)*.36,100*np.array(v)/n,color=color,width=.27,label=model.upper())
    b.bar_label(bars,labels=[f"{x}/{n}" for x in v],padding=4,fontsize=8)
b.set_title("Shared across both hosts and all layouts",loc="left")
b.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,frameon=False,fontsize=8)
fig.savefig(root/"second-host.png",dpi=170)
