"""Plot the matched repair and ablation counts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
counts = json.loads((root / "audit.json").read_text())["complete"]
arms = ("donor", "host_feature", "shuffled_donor", "intact_donor", "off", "native")
names = ("Aligned donor bridge", "Host-feature bridge", "Shuffled donor bridge",
         "Intact 7B donor", "Aligned bridge, donor off", "Untouched 14B host")
colors = ("#84dfc2", "#b6acf0", "#b6acf0", "#e8bc79", "#e8bc79", "#a7b4ae")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
bars = ax.barh(names, [counts[k] for k in arms], color=colors)
ax.invert_yaxis()
ax.set(title="The recipient uses the donor, and the host-only bridge nearly ties",
       xlabel="Complete repair groups out of 256", xlim=(0, 280))
ax.set_xticks([0, 64, 128, 192, 256])
ax.grid(axis="x", alpha=.18)
ax.bar_label(bars, padding=5)
fig.savefig(root / "repairs.png", dpi=170)
