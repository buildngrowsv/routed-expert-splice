"""Plot the fixed, routed and ablated feature outcomes."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
d=json.loads((here/"audit.json").read_text())["scores"]
arms=(("calibration_mean","Same calibration mean","#84dfc2"),("teacher","Full teacher features","#b6acf0"),
      ("aligned","Aligned block-10 features","#b6acf0"),("shuffled","Shuffled training targets","#e8bc79"),
      ("cyclic","Cyclic input swap","#e8bc79"),("host","Host-feature bridge","#a7b4ae"),("off","Donor ablated","#a7b4ae"))
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(8.2,4.4),facecolor="#101916",layout="constrained")
ax.set_facecolor("#101916")
bars=ax.barh([label for _,label,_ in arms],[d[k] for k,_,_ in arms],color=[c for _,_,c in arms])
ax.bar_label(bars,padding=4)
ax.invert_yaxis()
ax.grid(axis="x",alpha=.18)
ax.set(title="A constant input improved repairs inside the frozen recipient",
       xlabel="Complete groups out of 512",xlim=(0,265))
fig.savefig(here/"constant.png",dpi=170)
