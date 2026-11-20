"""Compare post-intervention log odds in the two checkpoints."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8.8,3.9),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
keys = ("native","matched","learned","whole_site")
labels = ("Untouched","Random directions, same norm","Learned 25 directions","Entire source token")
for off,name,label,color in ((-.15,"donor","Qwen coder 7B","#85dabc"),(.15,"host","Qwen general 14B","#a7b8d7")):
    vals = [d[name]["average_correct_log_odds"][key] for key in keys]
    bars = ax.barh([n+off for n in range(len(keys))],vals,height=.27,label=label,color=color)
    ax.bar_label(bars,fmt="%+.1f",padding=4)
ax.axvline(0,color="#d2dae3",lw=1)
ax.set(yticks=range(4),yticklabels=labels,xlim=(-16,16),xlabel="Log odds of the original letter versus the partner letter")
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.21),ncol=2,frameon=False)
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"subspace.png",dpi=165)
