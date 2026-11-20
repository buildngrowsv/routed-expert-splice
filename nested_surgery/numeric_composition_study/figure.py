"""Show full-panel native competence alongside eligible-group intervention counts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.4,4.1),layout="constrained",facecolor="#101916")
left,right=axes
for ax in axes:
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
bars=left.bar(range(5),d["native_groups_by_correct_items"],color=["#a7b8d7"]*4+["#85dabc"])
left.bar_label(bars,padding=3)
left.set(ylim=(0,840),xticks=range(5),xlabel="Correct variants out of four",ylabel="Native groups")
arms=("left","right","both","random","whole")
bars=right.barh(range(5),[d["cells"][m] for m in arms],color=["#a7b8d7"]*2+["#85dabc","#adb8b2","#a7b8d7"])
right.bar_label(bars,padding=4)
right.set(xlim=(0,10),yticks=range(5),yticklabels=("Left","Right","Both","Random","Full host states"),
          xlabel="Successful groups within the eight eligible")
right.invert_yaxis()
fig.savefig(root/"composition.png",dpi=165)
