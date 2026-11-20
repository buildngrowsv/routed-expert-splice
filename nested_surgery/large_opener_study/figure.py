from pathlib import Path
import json
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
models=[("0.5B coder","standalone-donor"),("1.5B general","pre-native"),("7B coder","donor"),("14B general","host")]
small=json.loads((here.parent/"opener_study/results/result.json").read_text())
large=json.loads((here/"results/result.json").read_text())
items=[small["scores"][k] if i<2 else large["individual"][k] for i,(_,k) in enumerate(models)]
trips=[small["triplets"][k] if i<2 else large["complete"][k] for i,(_,k) in enumerate(models)]
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#edf4ed",
                    "axes.labelcolor":"#e2ebe5","xtick.color":"#e2ebe5","ytick.color":"#e2ebe5","font.size":11,
                    "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(8.8,3.4),layout="constrained")
colors=["#74cbb0","#95a6bd"]*2
for ax,counts,n,title in zip(axes,[trips,items],[256,768],["All three repairs correct","Individual repairs correct"]):
    bars=ax.barh([m[0] for m in models],counts,color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars,labels=[str(x) for x in counts],padding=4)
    ax.set(xlim=(0,n*1.01),title=title,xlabel=f"Count out of {n}")
    ax.xaxis.grid(alpha=.13);ax.set_axisbelow(True)
fig.savefig(here/"qualification.png",dpi=165)
