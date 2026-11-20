"""Plot complete-group solves from a ten-block feature source and its controls."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
data=json.loads((here/"audit.json").read_text())["counts"]
names=(("intact","Intact 7B","#a7b4ae"),("teacher","Full 28-block features","#b6acf0"),
       ("mapped","Predicted from block 10","#84dfc2"),("shuffled","Shuffled map targets","#e8bc79"),
       ("host","Host features","#b6acf0"),("off","Donor ablated","#a7b4ae"))
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8,4.3),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
bars=ax.barh([b for _,b,_ in names],[data[k] for k,_,_ in names],color=[c for _,_,c in names])
ax.invert_yaxis()
ax.set(title="The early substitute tied the full graft and missed the shuffled-control gate",
       xlabel="Complete repair groups out of 512",xlim=(0,330))
ax.grid(axis="x",alpha=.18)
ax.bar_label(bars,padding=4)
fig.savefig(here/"early-exit.png",dpi=170)
