"""Plot same-panel shared-cohort outcomes of the 14B and 32B receivers."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())["shared"]
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.2,4.15),layout="constrained",facecolor="#101916",sharex=True)
modes=("transfer","paired","norm_matched","whole_site","borrowed")
labels=("Fitted one-pass","Paired donor","Equal-norm random","Full host state","Borrowed 14B map")
for ax,k,title in zip(axes,("14b","32b"),("14B receiver","32B receiver"),strict=True):
    ax.set_facecolor("#101916")
    selected=[i for i,m in enumerate(modes) if m in d["arms"][k]]
    values=[d["arms"][k][modes[i]] for i in selected]
    colors=["#85dabc" if i==0 else "#cbb2ef" if i==4 else "#a7b8d7" for i in selected]
    bars=ax.barh(selected,values,color=colors)
    ax.bar_label(bars,padding=4)
    ax.set(title=title,yticks=selected,yticklabels=[labels[i] for i in selected],
           xlim=(0,d["eligible"]*1.2),ylim=(4.7,-.7),xlabel=f'Paired reversals among {d["eligible"]} shared clean pairs')
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"scale.png",dpi=165)
