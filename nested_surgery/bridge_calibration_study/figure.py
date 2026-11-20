"""Plot paired effects for the native, original, and recalibrated inputs."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
effects = json.loads((root/"audit.json").read_text())["effects"]
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8,3.5),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
rows = (("native","Native 7B input","#a7b7d5"),("archived","Original 14B map","#efb480"),
        ("ordinary","Refit, ordinary","#83d8b9"),("weighted","Refit, weighted","#83d8b9"))
for i,(name,label,color) in enumerate(rows):
    p = effects[name]["mean"]
    lo,hi = effects[name]["interval"]
    ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt="o",color=color,capsize=5)
    ax.annotate(f"{p:+.2f}",(hi,i),xytext=(8,0),textcoords="offset points",va="center")
ax.axvline(0,color="#98a8b9",lw=1)
ax.set(yticks=range(4),yticklabels=[item[1] for item in rows],xlim=(-.45,2.46),
       xlabel="Matching-bracket change in log odds (95% paired intervals)")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"calibration.png",dpi=165)
