"""Select one direct-read layer on calibration states and test it on held-out programs."""
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
from identifier_baseline import SITES, encoded_batches, make_pairs, prefix
from identifier_binding import patch, programs
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

MODES=("unrestricted","blocked","only_selected","except_selected")
LATE_START=18


def selection_strength(model,rows,axis,cut):
    vectors=mx.array(axis)
    totals=np.zeros(len(model.model.layers)-LATE_START+1,dtype=np.float64)
    count=0
    for ids,source,end,_,_ in encoded_batches(rows):
        index=mx.arange(len(source))
        chosen=mx.array(np.flatnonzero(np.arange(len(source))%4<2))
        partner=mx.array(np.arange(len(source))^1)
        clean=prefix(model,ids,cut)
        original=clean[index,source].astype(mx.float32)
        edit=((original[partner]-original)@vectors)@vectors.T
        altered=patch(clean,source,original+edit)
        routes=masks(ids.shape[1],source,end)
        allowed=routes["unrestricted"]
        denied=routes["relay_only"]
        for i,layer in enumerate(model.model.layers[cut:],start=cut+1):
            clean_next=layer(clean,denied)
            edit_next=layer(altered,denied)
            if i>=LATE_START:
                ref=clean_next[index,end].astype(mx.float32)
                clean_shift=layer(clean,allowed)[index,end].astype(mx.float32)-ref
                edit_shift=layer(altered,allowed)[index,end].astype(mx.float32)-edit_next[index,end].astype(mx.float32)
                strength=mx.linalg.norm((edit_shift-clean_shift)[chosen],axis=1)
                strength/=mx.maximum(mx.linalg.norm(ref[chosen],axis=1),1e-6)
                totals[i-LATE_START]+=float(strength.sum().item())
            mx.eval(clean_next,edit_next)
            clean,altered=clean_next,edit_next
        count+=len(chosen)
    return totals/count


def schedules(width,source,end,*,cut,depth,selected):
    routes=masks(width,source,end)
    allowed=routes["unrestricted"]
    denied=routes["relay_only"]
    blocks=range(cut+1,depth+1)
    return dict(unrestricted=allowed,blocked=denied,
                only_selected=tuple(allowed if i==selected else denied for i in blocks),
                except_selected=tuple(denied if i==selected else allowed for i in blocks))


def summarize(scores):
    retained={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    eligible=np.logical_and.reduce(tuple(retained.values()))
    n=int(eligible.sum())
    report=dict(decision="unresolved",eligible=n,clean_groups={m:int(v.sum()) for m,v in retained.items()})
    if not n:
        return report
    flips={m:(scores[m+"_edited"][eligible]<0).all(2) for m in MODES}
    counts={m:v.sum(0).tolist() for m,v in flips.items()}
    first={m:v[:,0].astype(int) for m,v in flips.items()}
    draws=np.random.default_rng(107481).integers(n,size=(5000,n))
    only=interval(first["only_selected"]-first["unrestricted"],draws)
    removal=interval(first["unrestricted"]-first["except_selected"],draws)
    controls=(n>=100 and counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n
              and counts["blocked"][0]<=.15*n)
    sufficient=controls and counts["only_selected"][0]>=.7*n and only["interval"][0]>=-.15
    necessary=controls and removal["interval"][0]>=.2
    report.update(decision="isolated_edge" if sufficient or necessary else "unresolved",
                  reversals=counts,only_margin=only,removal_effect=removal,sufficient=sufficient,
                  necessary=necessary,controls_qualified=controls)
    return report


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    path=model_files("host")
    tok=Tokenizer.from_file(str(path/"tokenizer.json"))
    cal=make_pairs([tok],512,0)
    selection=programs(tok,64,104000)
    panel=programs(tok,384,105000)
    model,_=load_model(path,strict=True)
    cut=SITES["host"]
    depth=len(model.model.layers)
    assert cut==10 and depth==48
    letters=[ord(c)-97 for row in cal for c in row["letters"]]
    axis,_=bases(gather(model,cal,cut),letters)
    strengths=selection_strength(model,selection,axis,cut)
    assert np.isfinite(strengths).all()
    selected=int(strengths.argmax())+LATE_START
    print(f"Calibration selected block {selected}",flush=True)
    plan=partial(schedules,cut=cut,depth=depth,selected=selected)
    scores=evaluate(model,panel,axis,cut,modes=MODES,mask_factory=plan)
    assert all(v.shape==(384,2,2) and np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(out/"single-edge.npz",axis=axis,selection_strength=strengths,**scores)
    (out/"programs.json").write_text(json.dumps(dict(selection=selection,test=panel),separators=(",",":"))+"\n")
    report=dict(model=MODELS["host"],groups=len(panel)//2,selected_block=selected,
                selection_groups=len(selection)//2,cut=cut,depth=depth,**summarize(scores))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
