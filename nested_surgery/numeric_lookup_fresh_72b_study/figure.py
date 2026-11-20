"""Compare early and late donor effects in the discovery and replication cohorts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
first=json.loads((root.parent/"numeric_lookup_transfer_72b_study/audit.json").read_text())
fresh=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.7,3.8),layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
groups=[first["eligible"],fresh["eligible"]]
keys=("early_donor","late_donor","late_orthogonal","late_whole")
labels=("Early donor","Late donor","Late orthogonal","Late whole state")
colors=("#97a49c","#84dfc2","#e0b782","#b6acf0")
for i,(mode,label,color) in enumerate(zip(keys,labels,colors,strict=True)):
    v=[row["mirrored_switches"][mode] for row in (first,fresh)]
    bars=ax.bar(np.arange(2)+(i-1.5)*.193,100*np.array(v)/groups,width=.172,label=label,color=color)
    ax.bar_label(bars,labels=[f"{k}/{n}" for k,n in zip(v,groups,strict=True)],padding=4,fontsize=8)
ax.set_ylim(0,113)
ax.set_yticks([0,25,50,75,100],["0%","25%","50%","75%","100%"])
ax.set_xticks([0,1],["Original paired panel","Disjoint replication"])
ax.set_ylabel("Groups switching both mirrored lookups")
ax.set_axisbelow(True)
ax.grid(axis="y",alpha=.2)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncol=4,frameon=False,fontsize=8)
fig.savefig(root/"fresh-lookup.png",dpi=170)
