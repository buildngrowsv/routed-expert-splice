"""Render complete-sequence successes in the two digit-length directions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())["direction_counts"]
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.8,4.0),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
modes=("both","random","whole")
for i,(key,label,color) in enumerate((("one_to_two","One to two digits","#85dabc"),("two_to_one","Two to one digit","#a7b8d7"))):
    data=d[key]
    vals=[data["correct"][m] for m in modes]
    bars=ax.barh([j+(i-.5)*.31 for j in range(3)],vals,height=.27,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v}/{data["prompts"]}' for v in vals],padding=4)
ax.set(yticks=range(3),yticklabels=("Both donor edits","Equal-norm random","Full host states"),xlim=(0,900),
       xlabel="Complete numbers correct under full-vocabulary scoring")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.21),frameon=False,ncol=2)
fig.savefig(root/"carry.png",dpi=165)
