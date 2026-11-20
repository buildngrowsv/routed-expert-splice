"""Show full-vocabulary reversal counts on the one-pass comparison."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
v=json.loads((root/"audit.json").read_text())["full_vocabulary"]
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.7,3.6),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
modes=("transfer","paired","norm_matched","whole_site")
counts=[v["reversals"][m] for m in modes]
bars=ax.barh(range(4),counts,color=["#85dabc","#a7b8d7","#adb8b2","#a7b8d7"])
ax.bar_label(bars,labels=[f'{n}/{v["eligible"]}' for n in counts],padding=4)
ax.set(xlim=(0,585),yticks=range(4),yticklabels=("One donor pass","Paired donor states","Equal-norm random","Complete host state"),
       xlabel="Both counterfactual digits uniquely top-ranked")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"one-pass.png",dpi=165)
