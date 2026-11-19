from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/"results"
audit=json.loads((R/"audit.json").read_text())["choice_analysis"]
x=np.arange(4)
fig,ax=plt.subplots(figsize=(8.2,4.1),layout="constrained")
for j,(arm,name,color) in enumerate((("host","1.5B general host","#435b77"),("donor","0.5B coding donor","#267d70"))):
    bars=ax.bar(x+(.18 if j else -.18),audit[arm]["prediction_counts"],.35,color=color,label=name)
    ax.bar_label(bars,padding=2,fontsize=9)
ax.axhline(64,ls="--",color="#5e6264",label="Correct examples per indentation",lw=1)
ax.set_xticks(x,["4 spaces","8 spaces","12 spaces","16 spaces"])
ax.set_ylim(0,285);ax.set_ylabel("Model choices on 256 executable repairs")
ax.set_title("The host always chose the shallowest indentation",loc="left",fontsize=14,fontweight="bold")
ax.legend(frameon=False,fontsize=9);ax.spines[["top","right"]].set_visible(False)
fig.savefig(R/"indentation.png",dpi=170,bbox_inches="tight")

