"""Render the fresh-panel comparisons and donor-removal controls."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
audit = json.loads((here / "audit.json").read_text())
scores = audit["scores"]
rows = [
    ("Standalone 7B coder", audit["standalone_donor_only"], "#79d7b6"),
    ("Untouched 14B host", scores["before-native"]["groups"], "#8193a5"),
    ("Trained native host", scores["native"]["groups"], "#96aac6"),
    ("Trained coding graft", scores["coder"]["groups"], "#e9b677"),
    ("Trained shuffled graft", scores["shuffled"]["groups"], "#96aac6"),
    ("Coding graft switched off", scores["coder-to-knockout"]["groups"], "#a59ab9"),
]
assert scores["before-native"]["groups"] == 0
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
labels, values, palette = zip(*rows)
bars = ax.barh(labels, values, color=palette)
ax.bar_label(bars, padding=5)
ax.invert_yaxis()
ax.set_xlim(0, max(values) * 1.14)
ax.set_xlabel("Complete three-opener groups solved (out of 256)")
ax.grid(axis="x", alpha=.15)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(here / "complete-repairs.png", dpi=170)
