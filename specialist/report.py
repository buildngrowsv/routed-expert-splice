"""Aggregate full-path exchange, including cluster uncertainty and per-example changes."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)

def paired_interval(difference,groups):
    names=np.unique(groups)
    numerators=np.stack([difference[:,groups==f].sum(1) for f in names],axis=1)
    denominators=np.array([(groups==f).sum() for f in names])
    rng=np.random.default_rng(84971)
    draws=[]
    for _ in range(20):
        i=rng.integers(len(names),size=(250,1,len(names)))
        j=rng.integers(len(SEEDS),size=(250,len(SEEDS),1))
        draws.extend(numerators[j,i].sum((1,2))/(len(SEEDS)*denominators[i].sum((1,2))))
    return [float(difference.mean()),*np.quantile(draws,[.025,.975]).tolist()]

def main():
    panels={}
    for split in ("qualification","test"):
        rows=json.loads((R/f"{split}-problems.json").read_text())
        groups=np.array([r["family"] for r in rows])
        with np.load(R/f"{split}.npz") as data,np.load(R/f"{split}-live-exchange.npz") as ex:
            gold=data["gold"]
            correct=lambda z:z.argmax(1)==gold
            base=correct(data["donor_base"])
            donor=np.stack([correct(data[f"donor_{s}"]) for s in SEEDS])
            host=np.stack([correct(data[f"host_{s}"]) for s in SEEDS])
            shuffled=np.stack([np.mean([correct(ex[f"exchange{j}_{s}"]) for j in (1,2,3)],axis=0) for s in SEEDS])
            new=(~base)&donor
            lost=base&(~donor)
            summary=dict(n=len(rows),raw_donor=int(base.sum()),raw_host=int(correct(data["host_base"]).sum()),
                         donor=donor.sum(1).tolist(),host=host.sum(1).tolist(),gains=new.sum(1).tolist(),losses=lost.sum(1).tolist(),
                         gains_in_every_seed=int(new.all(0).sum()),losses_in_every_seed=int(lost.all(0).sum()),
                         family_and_seed_intervals={k:paired_interval(v,groups) for k,v in {
                            "donor_minus_host":donor.astype(int)-host,"donor_minus_base":donor.astype(int)-base,
                            "intact_minus_exchange":donor.astype(float)-shuffled}.items()},
                         shuffled_average_solves=shuffled.sum(1).tolist(),
                         donor_choice_counts=[np.bincount(data[f"donor_{s}"].argmax(1),minlength=4).tolist() for s in SEEDS],
                         unchanged_quartets=[int((p==p[:,:1]).all(1).sum()) for p in
                            [data[f"donor_{s}"].argmax(1).reshape(-1,4) for s in SEEDS]],
                         jointly_gained_examples=[r["id"] for r,flag in zip(rows,new.all(0)) if flag])
            panels[split]=summary
    (R/"summary.json").write_text(json.dumps(panels,indent=2)+"\n")
    for k,v in panels.items():print(k,{m:a for m,a in v.items() if m!="jointly_gained_examples"})
if __name__=="__main__":main()

