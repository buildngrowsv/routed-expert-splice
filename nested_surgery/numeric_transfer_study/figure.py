"""Render audited operand-reversal counts on a dark background."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())["by_operator"]
arms = ("transfer","norm_matched","whole_site")
labels = ("Translated 7B operand","Equal-norm random edit","Full 14B assignment state")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.4,3.9),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
for offset, op, color in ((-.16,"addition","#85dabc"),(.16,"subtraction","#a7b8d7")):
    counts = [d[op]["reversals"][k] for k in arms]
    runs = ax.barh([i+offset for i in range(3)],counts,height=.28,color=color,label=op.capitalize())
    ax.bar_label(runs,labels=[f'{n}/{d[op]["eligible"]}' for n in counts],padding=4)
ax.set(yticks=range(3),yticklabels=labels,xlim=(0,320),xlabel="Pairs in which both answers reversed")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.20),frameon=False,ncol=2)
fig.savefig(here/"operand.png",dpi=165)
