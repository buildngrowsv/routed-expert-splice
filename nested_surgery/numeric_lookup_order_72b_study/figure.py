"""Plot the causal switches and the separate change in native solves."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
run = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.55), layout="constrained")
fig.patch.set_facecolor("#101916")
for ax in axes:
    ax.set_facecolor("#101916")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0,111)
    ax.set_yticks([0,25,50,75,100], ["0%","25%","50%","75%","100%"])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.22)
colors = ("#b6acf0","#84dfc2")
left, right = axes
native = [run["native_early"],run["native_late"]]
bars = left.bar([0,1],100*np.array(native)/256,color=colors,width=.51)
left.bar_label(bars,labels=[f"{k}/256" for k in native],padding=4,fontsize=9)
left.set_title("Unedited model, all groups",loc="left")
left.set_xticks([0,1],["Key before table","Key after table"])
n=run["shared_native_groups"]
for i, order in enumerate(("early","late")):
    values = [run["whole"][order][k] for k in ("switch_both","inert")]
    boxes=right.bar(np.arange(2)+(i-.5)*.30,100*np.array(values)/n,color=colors[i],width=.25,
                    label=("Key before table","Key after table")[i])
    right.bar_label(boxes,labels=[f"{v}/{n}" for v in values],padding=4,fontsize=8)
right.set_title("Whole-state swap, shared native groups",loc="left")
right.set_xticks([0,1],["Both contexts switch","Constant table preserves"])
right.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=8)
fig.savefig(root/"key-order.png",dpi=170)
