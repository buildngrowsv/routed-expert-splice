"""Draw the matched changes and donor-withdrawal contrast from summary.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent / "results"
s = json.loads((R/"summary.json").read_text())
colors = ("#176658", "#a34d1e")
labels = ("Coding donor", "General donor", "Shuffled donor", "Direct affine", "Removed", "Native host")
arms = ("coder", "general", "shuffled", "direct", "knockout", "native")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios":[1.18, 1]})
fig.subplots_adjust(left=.14, right=.98, top=.79, bottom=.21, wspace=.52)
fig.suptitle("Can the host learn to use the transplant?", x=.04, y=.98, ha="left",
             fontsize=19, fontweight="bold")
for axis in (ax, bx):
    axis.spines[["top","right","left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.axvline(0, color="#555d61", lw=.8)
    axis.grid(axis="x", color="#e3e6e7", lw=.6)
    axis.set_axisbelow(True)
for j, (d,c) in enumerate(zip(("code","docs"), colors)):
    shift = (-.13,.13)[j]
    for axis, keys in ((ax, ["gain_"+a for a in arms]),
                       (bx, ["off_before", "off_after", "increased_use"])):
        for i, k in enumerate(keys):
            v = s["comparisons"][d][k]
            lo,hi = v["interval"]
            axis.hlines(i+shift, lo, hi, color=c, lw=2.2)
            axis.plot(v["mean"], i+shift, "o", color=c, ms=5.5)
ax.set_yticks(np.arange(6), labels)
bx.set_yticks(np.arange(3), ["Before training", "After training", "Change in reliance"])
for axis, title, xlabel in ((ax,"A   Gain from training block 28","Loss reduction (nats per token)"),
                            (bx,"B   Cost of removing the coding donor","Increase in loss (nats per token)")):
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=11, pad=13, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=10)
fig.legend([Line2D([],[],marker="o",lw=0,color=c) for c in colors],
           ["Python code","Documentation"], loc="lower left", bbox_to_anchor=(.035,.007),
           ncol=2, frameon=False, handletextpad=0, fontsize=10)
fig.text(.47,.055,"Intervals resample source files and paired training seeds (95%).", fontsize=9, color="#444")
fig.savefig(R/"donor-use.png",dpi=165,facecolor="white")
