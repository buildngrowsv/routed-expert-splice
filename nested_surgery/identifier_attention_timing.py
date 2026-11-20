"""Separate early and late availability of one declaration-to-readout edge."""
from functools import partial
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_attention_paths import evaluate, masks
from identifier_baseline import SITES, make_pairs
from identifier_binding import programs
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

MODES=("unrestricted","early","late","blocked")
BOUNDARY=17


def schedules(width,source,end,*,cut,depth,boundary=BOUNDARY):
    paths=masks(width,source,end)
    opened=paths["unrestricted"]
    closed=paths["relay_only"]
    n_early=boundary-cut
    n_late=depth-boundary
    assert n_early>0 and n_late>0
    return dict(unrestricted=opened,early=(opened,)*n_early+(closed,)*n_late,
                late=(closed,)*n_early+(opened,)*n_late,blocked=closed)


def summarize(scores):
    retained={k:(scores[k+"_clean"]>0).all(axis=(1,2)) for k in MODES}
    eligible=np.logical_and.reduce(tuple(retained.values()))
    n=int(eligible.sum())
    result=dict(decision="unresolved",eligible=n,clean_groups={k:int(x.sum()) for k,x in retained.items()})
    if not n:
        return result
    reversals={k:(scores[k+"_edited"][eligible]<0).all(2) for k in MODES}
    counts={k:v.sum(0).tolist() for k,v in reversals.items()}
    first={k:v[:,0].astype(int) for k,v in reversals.items()}
    draw=np.random.default_rng(741620).integers(n,size=(5000,n))
    deltas={m:interval(first[m]-first["unrestricted"],draw) for m in ("early","late")}
    late_first=interval(first["late"]-first["early"],draw)
    controls=(n>=100 and counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n
              and counts["blocked"][0]<=.15*n)
    sufficient=[m for m,d in deltas.items() if controls and counts[m][0]>=.7*n and d["interval"][0]>=-.15]
    lead=None
    if controls and late_first["interval"][0]>.2:
        lead="late"
    if controls and late_first["interval"][1]<-.2:
        lead="early"
    result.update(decision="timing_resolved" if lead or sufficient else "unresolved",
                  reversals=counts,relative_to_full=deltas,late_minus_early=late_first,
                  sufficient=sufficient,leading_window=lead,controls_qualified=controls)
    return result


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    path=model_files("host")
    tokenizer=Tokenizer.from_file(str(path/"tokenizer.json"))
    cal=make_pairs([tokenizer],512,0)
    panel=programs(tokenizer,384,98000)
    model,_=load_model(path,strict=True)
    cut=SITES["host"]
    depth=len(model.model.layers)
    assert cut==10 and depth==48
    labels=[ord(c)-97 for row in cal for c in row["letters"]]
    axis,_=bases(gather(model,cal,cut),labels)
    factory=partial(schedules,cut=cut,depth=depth)
    scores=evaluate(model,panel,axis,cut,modes=MODES,mask_factory=factory)
    assert all(v.shape==(384,2,2) and np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(out/"timing.npz",axis=axis,**scores)
    (out/"programs.json").write_text(json.dumps(panel,separators=(",",":"))+"\n")
    result=dict(model=MODELS["host"],groups=len(panel)//2,boundary=BOUNDARY,cut=cut,depth=depth,**summarize(scores))
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
