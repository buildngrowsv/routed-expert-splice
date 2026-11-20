"""Show how the same activation swap changes individual and whole-group answers."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
data=json.loads((here/"audit.json").read_text())
keys=("relocated_definition","alias_copy","lookup_read","close_bracket","joint")
labels=("Relocated definition","Copy assignment","Alias in lookup","Closing bracket","Both lookup tokens")
colors=("#84dfc2","#a7b4ae","#b6acf0","#b6acf0","#e8bc79")
plt.style.use("dark_background")
fig,axs=plt.subplots(1,2,figsize=(9.3,4.0),sharey=True,facecolor="#101916",layout="constrained")
arms=([data["individual_switches"][k] for k in keys],[data["counts"][k]["switch"] for k in keys])
for ax,vals,title,xlim in zip(axs,arms,("Individual switches out of 572","Complete switches out of 143"),(605,154),strict=True):
    ax.set_facecolor("#101916")
    ax.grid(axis="x",alpha=.18)
    bars=ax.barh(labels,vals,color=colors)
    ax.bar_label(bars,padding=4)
    ax.set(title=title,xlim=(0,xlim))
axs[0].invert_yaxis()
fig.savefig(here/"alias-use.png",dpi=170)
