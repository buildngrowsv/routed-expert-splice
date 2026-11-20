"""Show the borrowed-direction magnitude contrast on the shared cohort."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
with np.load(root/"results/raw-32b.npz") as x, np.load(root.parent/"numeric_scale_study/results/32b.npz") as y:
    ratios=x["norms"][:,0]/y["norms"][:,3]
plt.style.use("dark_background")
fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout="constrained",facecolor="#101916")
left,right=axes
for ax in axes:
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
modes=("borrowed_raw","norm_matched","borrowed_scaled")
bars=left.barh(range(3),[d["reversals"][m] for m in modes],color=["#85dabc","#adb8b2","#a7b8d7"])
left.bar_label(bars,padding=4)
left.set(xlim=(0,507),yticks=range(3),yticklabels=("Raw 14B","Equal-norm random","Norm-matched 14B"),
         xlabel="Full-vocabulary paired reversals, out of 507")
left.invert_yaxis()
right.hist(ratios,bins=25,color="#85dabc")
right.axvline(1,color="#d2dae3",ls="--",lw=1)
right.set(xlabel="Raw edit norm divided by matched norm",ylabel="Source tokens")
fig.savefig(root/"raw-32b.png",dpi=165)
