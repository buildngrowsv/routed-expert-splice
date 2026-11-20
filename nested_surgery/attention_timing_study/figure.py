"""Show which layers allowed a direct read and how often the edit reversed a pair."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(9.1,3.9),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
modes=("unrestricted","early","late","blocked")
labels=("Throughout","Through block 17","After block 17","Closed")
spans=((11,48),(11,17),(18,48),None)
for i,(m,span) in enumerate(zip(modes,spans,strict=True)):
    ax.barh(i,38,left=10.5,height=.52,color="#30433c")
    if span is not None:
        a,b=span
        ax.barh(i,b-a+1,left=a-.5,height=.52,color="#85dabc")
    ax.text(50,i,f'{d["reversals"][m][0]}/384',ha="left",va="center",color="#e5eee9")
ax.axvline(17.5,color="#d2dae3",lw=.8,ls="--")
ax.set(yticks=range(4),yticklabels=labels,xticks=[11,17,25,33,41,48],xlim=(10,57),xlabel="Transformer block")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.text(50,-.56,"Paired reversals",fontsize=10,ha="left")
fig.savefig(root/"edge-timing.png",dpi=165)
