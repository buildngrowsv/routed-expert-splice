"""Plot the route's fresh complete-repair counts at different call costs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--confirmation", action="store_true")
confirmation = parser.parse_args().confirmation
if confirmation:
    here = here.parent / "frozen_graft_route_confirmation_study"
audit = json.loads((here / "audit.json").read_text())
result = json.loads((here / "results/result.json").read_text())
groups = result["groups"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
specs = [("Intact 7B", "#b6acf0"), ("Random calls, same budget", "#e8bc79"),
         ("Confidence blend", "#84dfc2"), ("Always blend", "#a7b4ae")]
for i, (label, color) in enumerate(specs):
    values = []
    for layout in ("neutral", "decoys"):
        r = audit[layout]
        values.append(r["mean_random_complete"] if i == 1 else
                      r["complete"][{0:"intact", 2:"confidence_blend", 3:"always_blend"}[i]])
    bars = ax.bar(np.arange(2)+(i-1.5)*.18, values, width=.17, label=label, color=color)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize=8, padding=3)
ax.set_xticks([0, 1], ["Neutral controls", "Bracket decoys"])
title = "Confidence routing on fresh neutral and bracket-decoy repairs" if confirmation else (
    "Confidence routing helped on decoys and narrowly missed the neutral guard")
ax.set(title=title, ylabel=f"Complete groups out of {groups:,}", ylim=(0, groups*1.07))
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.grid(axis="y", alpha=.18)
fig.savefig(here / "route.png", dpi=170)
