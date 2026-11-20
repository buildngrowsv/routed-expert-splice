"""Plot calibration strength and observed held-out behavior."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.5,3.9),layout="constrained",facecolor="#101916")
a,b=axes
for ax in axes:
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
blocks=list(range(18,49))
a.bar(blocks,d["calibration_strengths"],color=["#85dabc" if k==d["selected_block"] else "#a7b8d7" for k in blocks])
a.set(xticks=[18,20,28,38,48],xlabel="Transformer block",ylabel="Normalized calibration displacement")
modes=("unrestricted","except_selected","only_selected","blocked")
labels=("All blocks","All except block 20","Only block 20","Closed")
bars=b.barh(range(4),[d["reversals"][k][0] for k in modes],color="#85dabc")
b.bar_label(bars,padding=4)
b.set(xlim=(0,410),yticks=range(4),yticklabels=labels,xlabel="Held-out paired reversals out of 384")
b.invert_yaxis()
fig.savefig(root/"single-edge.png",dpi=165)
