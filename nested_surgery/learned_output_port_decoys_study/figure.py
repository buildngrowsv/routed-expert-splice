"""Plot complete held-out groups for matched neutral and decoy layouts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
audit = json.loads((here / "audit.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.8, 4.5), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
labels = ("Intact specialist", "Aligned graft", "Host-feature bridge", "Shuffled graft", "Donor off")
keys = ("intact_donor", "donor", "host_feature", "shuffled_donor", "off")
x = np.arange(len(keys))
for layout, offset, color, label in (("neutral", -.18, "#b6acf0", "Neutral text"),
                                      ("decoys", .18, "#84dfc2", "Bracket decoys")):
    bars = ax.bar(x+offset, [audit[layout]["complete"][k] for k in keys], .33, color=color, label=label)
    ax.bar_label(bars, padding=3, fontsize=9)
ax.set(title="The graft gained an advantage under bracket decoys", ylabel="Complete groups out of 256", ylim=(0, 255))
ax.set_xticks(x, labels)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.18)
fig.savefig(here / "decoys.png", dpi=170)
