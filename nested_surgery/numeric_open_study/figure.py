"""Plot exact counterfactual pair reversals by scoring rule."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
result=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(9,4),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
modes=("transfer","norm_matched","whole_site")
rows=(
    ("full_vocabulary","Complete vocabulary","#85dabc"),
    ("forced_two_choice","Two answer tokens","#a7b8d7"),
)
for i,(key,name,color) in enumerate(rows):
    d=result[key]
    vals=[d["reversals"][m] for m in modes]
    bars=ax.barh([j+(i-.5)*.32 for j in range(3)],vals,height=.28,label=name,color=color)
    ax.bar_label(bars,labels=[f'{x}/{d["eligible"]}' for x in vals],padding=4)
ax.set(yticks=range(3),yticklabels=("Translated donor","Equal-norm random","Full host state"),xlim=(0,600),
       xlabel="Pairs with both counterfactual predictions correct")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.22),ncol=2,frameon=False)
fig.savefig(root/"open-vocabulary.png",dpi=165)
