"""Show the reused calibration ranking and held-out outcomes of both sets."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
with np.load(root/"results/sparse-edges.npz") as data:
    z=data["selection_strength"]
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.6,4.2),layout="constrained",facecolor="#101916")
a,b=axes
for ax in axes:
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
strong,weak="#85dabc","#cbb2ef"
blocks=list(range(18,49))
a.bar(blocks,z,color=[strong if j in d["strong"] else weak if j in d["weak"] else "#687872" for j in blocks])
for line in (25.5,33.5,41.5):
    a.axvline(line,color="#707e78",lw=.65,ls="--")
a.set(xticks=[18,25,33,41,48],xlabel="Transformer block",ylabel="Normalized calibration displacement")
a.legend(handles=[Patch(color=strong,label="Strong"),Patch(color=weak,label="Depth-matched weak")],
         loc="upper center",bbox_to_anchor=(.5,-.23),frameon=False,ncol=2)
m=("unrestricted","strong_eight","weak_eight","without_strong","blocked")
labels=("All blocks","Strong eight","Weak eight","All except strong","Closed")
bars=b.barh(range(len(m)),[d["reversals"][k][0] for k in m],color=["#a7b8d7",strong,weak,"#a7b8d7","#a7b8d7"])
b.bar_label(bars,padding=4)
b.set(xlim=(0,440),yticks=range(len(m)),yticklabels=labels,xlabel="Edited-class paired reversals out of 384")
b.invert_yaxis()
fig.savefig(root/"sparse-edges.png",dpi=165)
