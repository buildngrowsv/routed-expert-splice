"""Run only the frozen discovery screen before opening downstream panels."""
import json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"task_transfer"))
from screen import encode,full_logits,load_model
# task_transfer also has a problems module, so qualify the local generator.
sys.path.insert(0,str(HERE.parent))
from indentation.problems import PATCHES,panel

R=HERE/"results"

def compare(rows,host,donor):
    gold=np.array([r["gold"] for r in rows])
    h=host.argmax(1)==gold;d=donor.argmax(1)==gold
    groups=sorted(set(r["family"] for r in rows))
    g=np.array([r["family"] for r in rows])
    effects=np.array([d[g==k].sum()-h[g==k].sum() for k in groups])
    counts=np.array([(g==k).sum() for k in groups])
    picks=np.random.default_rng(8471).integers(len(groups),size=(5000,len(groups)))
    ci=np.quantile(effects[picks].sum(1)/counts[picks].sum(1),[.025,.975])
    return dict(n=len(rows),families=len(groups),host=int(h.sum()),donor=int(d.sum()),donor_only=int((d&~h).sum()),
        host_only=int((h&~d).sum()),donor_minus_host=float(d.mean()-h.mean()),family_interval=ci.tolist(),
        host_full_quartets=int(h.reshape(-1,4).all(1).sum()),donor_full_quartets=int(d.reshape(-1,4).all(1).sum()),
        families_raw={k:dict(problems=int((g==k).sum()),host=int(h[g==k].sum()),donor=int(d[g==k].sum())) for k in groups})

def main():
    R.mkdir(exist_ok=True)
    rows=panel("discovery")
    encoded=encode(rows,PATCHES)
    scores={};elapsed={};start=time.perf_counter()
    for name in ("host","donor"):
        model=load_model(name)
        scores[name]=full_logits(model,encoded)
        elapsed[name]=time.perf_counter()-start
        del model
        start=time.perf_counter()
    report=compare(rows,scores["host"],scores["donor"])
    report.update(discovery_qualified=report["donor_minus_host"]>=.05,inference_and_load_seconds=elapsed,choice_ids=encoded[2].tolist())
    np.savez_compressed(R/"discovery.npz",gold=encoded[-1],**scores)
    (R/"problems.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    (R/"discovery.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)

if __name__=="__main__":main()

