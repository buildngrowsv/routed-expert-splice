"""Evaluate two equally timed sets of source-to-readout attention edges."""
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_attention_paths import evaluate,masks
from identifier_baseline import SITES,make_pairs
from identifier_binding import programs
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS,model_files

MODES=("unrestricted","blocked","strong_eight","weak_eight","without_strong")
PRIOR="qwen-identifier-selected-edge"
REVISION="3e428e7a2e955398056e38a56193e42f4877b579"


def choose(strength):
    assert strength.shape==(31,) and np.isfinite(strength).all()
    high,low=[],[]
    for a,b in ((18,26),(26,34),(34,42),(42,49)):
        options=list(range(a,b))
        selected=sorted(options,key=lambda i:(-strength[i-18],i))[:2]
        comparison=sorted(set(options)-set(selected),key=lambda i:(strength[i-18],i))[:2]
        high.extend(selected)
        low.extend(comparison)
    return tuple(sorted(high)),tuple(sorted(low))


def schedules(width,source,end,*,cut,depth,strong,weak):
    paths=masks(width,source,end)
    opened,closed=paths["unrestricted"],paths["relay_only"]
    def across(chosen):
        return tuple(opened if i in chosen else closed for i in range(cut+1,depth+1))
    return dict(unrestricted=opened,blocked=closed,strong_eight=across(strong),weak_eight=across(weak),
                without_strong=across(set(range(cut+1,depth+1))-set(strong)))


def summarize(scores):
    clean={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    eligible=np.logical_and.reduce(tuple(clean.values()))
    n=int(eligible.sum())
    result=dict(decision="unresolved",eligible=n,clean_groups={m:int(v.sum()) for m,v in clean.items()})
    if not n:
        return result
    flips={m:(scores[m+"_edited"][eligible]<0).all(axis=2) for m in MODES}
    counts={m:v.sum(0).tolist() for m,v in flips.items()}
    primary={m:v[:,0].astype(int) for m,v in flips.items()}
    draws=np.random.default_rng(416083).integers(n,size=(5000,n))
    def diff(a,b):
        return interval(primary[a]-primary[b],draws)
    controls=(n>=100 and counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n
              and counts["blocked"][0]<=.15*n)
    relative={m:diff(m,"unrestricted") for m in ("strong_eight","weak_eight")}
    sufficient=[m for m,x in relative.items() if controls and counts[m][0]>=.7*n and x["interval"][0]>=-.15]
    high_low=diff("strong_eight","weak_eight")
    removal=diff("unrestricted","without_strong")
    ahead=controls and high_low["interval"][0]>=.2
    needed=controls and removal["interval"][0]>=.2
    result.update(decision="sparse_attention_resolved" if sufficient or ahead or needed else "unresolved",
                  reversals=counts,relative=relative,strong_minus_weak=high_low,removal=removal,
                  sufficient=sufficient,strong_leads=ahead,removal_material=needed,controls_qualified=controls)
    return result


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    previous=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/PRIOR
    receipt=json.loads((previous/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]==REVISION
    source=(previous/"single-edge.npz")
    with np.load(source) as data:
        strength=data["selection_strength"]
    high,low=choose(strength)
    print(f"Strong edges {high}. Matched weak edges {low}.",flush=True)
    path=model_files("host")
    tok=Tokenizer.from_file(str(path/"tokenizer.json"))
    cal=make_pairs([tok],512,0)
    panel=programs(tok,384,116000)
    model,_=load_model(path,strict=True)
    cut=SITES["host"]
    depth=len(model.model.layers)
    assert cut==10 and depth==48
    y=[ord(c)-97 for r in cal for c in r["letters"]]
    axis,_=bases(gather(model,cal,cut),y)
    schedule=partial(schedules,cut=cut,depth=depth,strong=high,weak=low)
    scores=evaluate(model,panel,axis,cut,modes=MODES,mask_factory=schedule)
    assert all(v.shape==(384,2,2) and np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(out/"sparse-edges.npz",axis=axis,selection_strength=strength,**scores)
    (out/"programs.json").write_text(json.dumps(panel,separators=(",",":"))+"\n")
    report=dict(model=MODELS["host"],groups=len(panel)//2,strong=high,weak=low,
                calibration_digest=sha256(source.read_bytes()).hexdigest(),**summarize(scores))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
