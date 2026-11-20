"""Plot selected-source and control reversals with the original denominators."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
data = json.loads((here/"audit.json").read_text())
modes = ("selected","unselected","whole_site")
labels = ("25 directions at selected class","Same edit at unselected class","Full state at selected class")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.9,3.8), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for offset, query, label, color in ((-.16,0,"Read first class","#85dabc"),(.16,1,"Read second class","#a7b8d7")):
    values = [data["reversals_by_site"][mode][query] for mode in modes]
    bars = ax.barh([i+offset for i in range(3)],values,height=.28,label=label,color=color)
    ax.bar_label(bars,padding=4)
ax.set(yticks=range(3), yticklabels=labels, xlim=(0,286), xlabel="Paired reads reversed out of 256")
ax.invert_yaxis()
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.23),ncol=2,frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"binding.png",dpi=165)
