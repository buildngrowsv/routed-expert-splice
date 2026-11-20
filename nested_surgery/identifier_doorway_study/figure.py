"""Compare the first-class answer margins across both relay panels."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
plt.style.use("dark_background")
fig, axes = plt.subplots(1,2,figsize=(10.5,4.5),layout="constrained",facecolor="#101916")
arms = (
    ("Assignment signs","audit.json",("same_class_relay","other_class_relay"),("Same class","Other class")),
    ("Final query token","query-audit.json",("query_capture","query_direct"),("Capture","Restricted")),
)
for ax,(title,file,keys,labels) in zip(axes,arms,strict=True):
    ax.set_facecolor("#101916")
    values = json.loads((here/file).read_text())["mean_original_letter_margin"]
    order = ("native","unrestricted",*keys)
    bars = ax.barh(range(4),[values[k][0] for k in order],color=["#a7b8d7","#85dabc","#adb8b2","#adb8b2"])
    ax.bar_label(bars,fmt="%+.2f",padding=4)
    ax.axvline(0,color="#d2dae3",lw=.8)
    ax.set(title=title,yticks=range(4),yticklabels=("Native","Unrestricted",*labels),
           xlim=(-7.5,16.5),xlabel="Mean logit margin for the original letter")
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"relay.png",dpi=165)
