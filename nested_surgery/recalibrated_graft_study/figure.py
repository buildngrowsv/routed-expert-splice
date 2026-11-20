"""Plot complete program-group repairs under the fixed training budget."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())["complete"]
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8,4),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
labels = ["Standalone 7B coder","Trained 14B host","Graft with original map","Graft with refitted map",
          "Shuffled graft, original map","Shuffled graft, refitted map"]
keys = ["standalone-donor","native","coder","refit","shuffled","refit-shuffled"]
counts = [d[k] for k in keys]
bars=ax.barh(labels,counts,color=["#86ddbd","#9cacc9","#9cacc9","#edbb85","#9cacc9","#9cacc9"])
ax.bar_label(bars,padding=5)
ax.set(xlim=(0,max(counts)*1.17+1),xlabel="Complete repair groups solved (out of 256)")
ax.invert_yaxis()
ax.grid(axis="x",alpha=.12)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"refitted-graft.png",dpi=165)
