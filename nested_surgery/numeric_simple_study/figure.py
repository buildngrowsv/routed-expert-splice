"""Plot qualified composition counts for each operation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())["by_operator"]
arms=("left","right","both","random","whole")
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.5,4.1),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
for i,(op,color) in enumerate((("addition","#85dabc"),("subtraction","#a7b8d7"))):
    bars=ax.barh([j+(i-.5)*.32 for j in range(5)],[d[op]["cells"][k] for k in arms],
                 height=.28,label=op.capitalize(),color=color)
    ax.bar_label(bars,padding=4)
ax.set(xlim=(0,310),yticks=range(5),yticklabels=("Left donor","Right donor","Both donors","Random edits","Full host states"),
       xlabel="Groups with all four outcomes correct, out of 256")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.20),frameon=False,ncol=2)
fig.savefig(root/"two-operands.png",dpi=165)
