"""Compare whole-triplet correctness under final-module interventions."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
scores = json.loads((here / "audit.json").read_text())["group_scores"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.9, 3.7), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
labels = ["Original output", "Output set to zero", "Borrowed · same opener", "Borrowed · changed opener"]
values = [scores[k] for k in ("intact", "off", "same", "changed")]
bars = ax.barh(labels, values, color=["#a4b2cf","#8d9892","#81d8ba","#e6b17c"])
ax.bar_label(bars, padding=5)
ax.invert_yaxis()
ax.set_xlim(0, max(values)*1.17+1)
ax.set_xlabel("Complete Python repair groups solved (out of 256)")
ax.grid(axis="x", alpha=.15)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here / "donor-exchange.png", dpi=165)
