"""Compare complete-vocabulary and four-choice outcomes on the same groups."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.9,4.2),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
modes=("native","both","random","whole")
for i,(key,label,color) in enumerate((("full_vocabulary","Complete vocabulary","#85dabc"),
                                     ("forced_four_choice","Four choices","#a7b8d7"))):
    counts=[d[key]["correct"][k] for k in modes]
    bars=ax.barh([j+(i-.5)*.31 for j in range(4)],counts,height=.27,label=label,color=color)
    ax.bar_label(bars,padding=4)
ax.set(yticks=range(4),yticklabels=("Original host","Both donor edits","Equal-norm random","Full host states"),
       xlim=(0,600),xlabel="Groups with all four predictions correct, out of 512")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.21),ncol=2,frameon=False)
fig.savefig(root/"composition-open.png",dpi=165)
