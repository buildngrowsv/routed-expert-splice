"""Plot native retention and shared-cohort reversal comparisons."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.6,4),layout="constrained",facecolor="#101916")
modes=("unrestricted","blocked","strong_eight","weak_eight","without_strong")
labels=("All blocks","Direct edge closed","Strong eight only","Weak eight only","All except strong")
for ax,key,denominator,title,color in (
        (axes[0],"clean_groups",512,"Native answers","#a7b8d7"),
        (axes[1],"reversals",456,"Counterfactual answers","#85dabc")):
    ax.set_facecolor("#101916")
    values=[d[key][m] for m in modes]
    bars=ax.barh(range(5),values,color=color)
    ax.bar_label(bars,labels=[f"{v}/{denominator}" for v in values],padding=4)
    ax.set(title=title,xlim=(0,denominator*1.22),yticks=range(5),yticklabels=labels,xlabel="Both variants correct")
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"numeric-attention.png",dpi=165)
