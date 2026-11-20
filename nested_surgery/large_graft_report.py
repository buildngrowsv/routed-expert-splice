from pathlib import Path
import numpy as np


def summarize(out: Path, rows):
    names = ["native","knockout","direct","shuffled","general","coder"]
    names += ["before-"+k for k in names] + ["coder-to-knockout","coder-to-native","coder-to-shuffled"]
    gold = np.array([r["gold"] for r in rows])
    individual = {k:np.load(out/(k+".npy")).argmax(1)==gold for k in names}
    trip = {k:v.reshape(-1,3).all(1) for k,v in individual.items()}
    idx = np.random.default_rng(688431).integers(len(gold)//3,size=(5000,len(gold)//3))
    comparisons={}
    for k in ("native","shuffled","general","direct","coder-to-knockout","coder-to-native"):
        delta = trip["coder"].astype(float)-trip[k]
        comparisons[k]=dict(gap=float(delta.mean()),interval=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
    new=trip["coder"] & ~trip["native"] & ~trip["before-native"]
    strict=new & ~trip["coder-to-knockout"] & ~trip["coder-to-native"]
    strong=all(comparisons[k]["gap"]>=.05 and comparisons[k]["interval"][0]>0 for k in ("native","shuffled"))
    def identifiers(mask):
        return [rows[3*i]["id"].rsplit("-",1)[0] for i in np.flatnonzero(mask)]
    return dict(decision="replicate" if strong and strict.sum()>=10 else "stop",groups=len(gold)//3,
        individual={k:int(v.sum()) for k,v in individual.items()},complete={k:int(v.sum()) for k,v in trip.items()},
        advantages=comparisons,new_ids=identifiers(new),strict_ids=identifiers(strict),
        native_regressions=identifiers(trip["native"] & ~trip["coder"]),
        off_rescues=identifiers(trip["coder-to-knockout"] & ~trip["coder"]),
        off_losses=identifiers(trip["coder"] & ~trip["coder-to-knockout"]))
