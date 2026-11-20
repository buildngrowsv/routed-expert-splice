"""Plot the alias and definition exchanges on the same native-correct programs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
report = json.loads((here / "results/result.json").read_text())
paired = report["cross_host_shared"]
count = paired["at_32b"]["shared"]
positions = np.arange(2)
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
if count:
    for shift, key, name, color in ((-.17, "alias", "Late copy", "#b6acf0"),
                                   (.17, "relocated", "Relocated declaration", "#84dfc2")):
        bars = ax.bar(positions + shift, [paired[k]["switched"][key] for k in ("at_32b", "at_72b")],
                      width=.3, label=name, color=color)
        ax.bar_label(bars, padding=3)
    ax.set_xticks(positions, ["32B after block 32", "72B after block 40"])
    ax.set(title="Midpoint swaps on the same native-correct lookups",
           ylabel=f"Complete switches out of {count}", ylim=(0, count*1.15))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(here / "midpoints.png", dpi=170)
