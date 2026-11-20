"""Show the matched bracket signal across the donor and host receiving paths."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())["effects"]
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8.6,3.45),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
keys = ("donor_native","host_native","host_exact_donor","host_refit_donor")
labels = ("Native 7B donor","Native 14B module","Exact donor output → 14B","Refitted donor output → 14B")
for i,k in enumerate(keys):
    p = d[k]["mean"]
    lo,hi = d[k]["interval"]
    ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt="o",color="#89d9bf" if i==0 else "#eec18c",capsize=5)
    ax.annotate(f"{p:+.3f}",(hi,i),xytext=(8,0),textcoords="offset points",va="center")
ax.axvline(0,color="#a7afc4",lw=1)
ax.set(yticks=range(4),yticklabels=labels,xlim=(-.19,2.59),
       xlabel="Matching-bracket change in log odds (95% paired intervals)")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"output-boundary.png",dpi=165)
