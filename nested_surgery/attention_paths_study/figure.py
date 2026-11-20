"""Compare paired reversals and matched clean/edited logit margins."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/"audit.json").read_text())
keys=("unrestricted","direct_only","relay_only")
labels=("Unrestricted","Direct edge only","Relay paths only")
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout="constrained",facecolor="#101916")
left,right=axes
for ax in axes:
    ax.set_facecolor("#101916")
    ax.set(yticks=range(3),yticklabels=labels)
    ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()
bars=left.barh(range(3),[data["reversals_by_query"][k][0] for k in keys],color="#85dabc")
left.bar_label(bars,padding=4)
left.set(xlim=(0,595),xlabel="First-class paired reversals out of 512")
for i,(arm,name,color) in enumerate((("clean","Clean declaration","#a7b8d7"),("edited","Edited declaration","#85dabc"))):
    bars=right.barh([j+(i-.5)*.32 for j in range(3)],[data["mean_margin"][f"{k}_{arm}"][0] for k in keys],
                    height=.28,color=color,label=name)
    right.bar_label(bars,fmt="%+.1f",padding=4)
right.axvline(0,color="#d2dae3",lw=.8)
right.set(xlim=(-7.8,17.4),xlabel="Mean logit margin for the original letter")
right.legend(loc="upper center",bbox_to_anchor=(.5,-.22),frameon=False,ncol=2)
fig.savefig(root/"attention-paths.png",dpi=165)
