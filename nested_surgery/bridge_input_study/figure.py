"""Plot direction and magnitude of the crossed candidate effect."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
audit = json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8,3.6),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
pairs = ["Parenthesis / square bracket","Parenthesis / brace","Square bracket / brace"]
for k,label,color,shift in (("native","Native 7B input","#87dec0",-.15),("mapped","Mapped 14B input","#f0b884",.15)):
    for i,p in enumerate(audit["pair_means"][k]):
        ax.barh(i+shift,p,height=.28,color=color,label=label if i==0 else None)
ax.axvline(0,color="#9caec4",lw=1)
ax.set(yticks=range(3),yticklabels=pairs,xlim=(-.71,2.67),xlabel="Matching-source change in log odds")
ax.invert_yaxis()
ax.legend(frameon=False,loc="upper right",fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"crossed-input.png",dpi=165)
