"""Show the single-gain outcome alongside both historical and random controls."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
d=json.loads((r/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.9,3.9),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
m=("raw","gain","fitted_magnitude","random")
labels=("Original 14B norm","Answer-free gain","Fitted 32B magnitude","Equal-norm random")
v=[d["reversals"][k] for k in m]
bars=ax.barh(range(4),v,color=["#a7b8d7","#85dabc","#a7b8d7","#adb8b2"])
ax.bar_label(bars,labels=[f'{n}/{d["eligible"]}' for n in v],padding=4)
ax.set(xlim=(0,585),yticks=range(4),yticklabels=labels,xlabel="Both counterfactual digits uniquely top-ranked")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(r/"scalar.png",dpi=165)
