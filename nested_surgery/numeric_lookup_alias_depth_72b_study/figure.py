"""Plot complete lookup switches on the native-shared panel."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / "results/result.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.9, 4.5), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
styles = (("relocated", "Relocated declaration", "#84dfc2", "s", "-"),
          ("source", "Original numeral in alias prompts", "#e8bc79", "o", "--"),
          ("alias", "Late alias read", "#b6acf0", "D", ":"))
for k, label, color, point, linestyle in styles:
    data = [result["metrics"][f"{k}_{cut}"]["switch"] for cut in result["cuts"]]
    ax.plot(result["cuts"], data, label=label, color=color, marker=point, linestyle=linestyle, lw=2)
ax.set(title="The late alias stayed inert through the informative midpoint",
       xlabel="Blocks completed before the swap", ylabel=f"Complete switches out of {result['shared']}",
       ylim=(-5, 150))
ax.set_xticks(result["cuts"])
ax.grid(axis="y", alpha=.18)
ax.legend(frameon=False, loc="upper right")
fig.savefig(root / "depth.png", dpi=170)
