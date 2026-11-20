"""Compare one-donor-pass projected state replacement with a paired activation edit."""
import gc
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES,encoded_batches,prefix
from identifier_computation import DIGITS,programs
from identifier_numeric_open import metrics,whole_vocabulary
from identifier_transfer import fit_map,gather,transform
from large_opener import MODELS,model_files

MODES=("native","transfer","norm_matched","paired","whole_site")


def fit_absolute(donor,host,axes):
    x=donor@axes[0]
    y=host@axes[1]
    xc=x-x.mean(0)
    yc=y-y.mean(0)
    gram=xc.T@xc
    penalty=.02*np.trace(gram)/len(gram)
    weights=np.linalg.solve(gram+penalty*np.eye(len(gram)),xc.T@yc)
    bias=y.mean(0)-x.mean(0)@weights
    return weights.astype(np.float32),bias.astype(np.float32)


def evaluate(host,rows,states,axes,fit,*,cut=SITES["host"],borrowed=None):
    donor_axis,host_axis,random_axis,_=axes
    weights,bias=fit
    scores={name:[] for name in (MODES if borrowed is None else (*MODES,"borrowed"))}
    doses=[]
    for i,(ids,source,end,choices,_) in enumerate(encoded_batches(rows)):
        hidden=prefix(host,ids,cut)
        index=mx.arange(len(source))
        partner=mx.array(np.arange(len(source))^1)
        original=hidden[index,source].astype(mx.float32)
        donor=mx.array(states[i*16:i*16+len(source)].astype(np.float32))
        target=donor[partner]@donor_axis@weights+bias
        change=target-original@host_axis
        single=change@host_axis.T
        random=change@random_axis.T
        wanted=mx.linalg.norm(single,axis=1,keepdims=True)
        actual=mx.linalg.norm(random,axis=1,keepdims=True)
        assert bool(mx.all((actual>0)|(wanted==0)).item())
        random=random*(wanted/mx.maximum(actual,1e-12))
        paired,_=transform(donor,*axes)
        place=(mx.arange(ids.shape[1])[None,:]==source[:,None])[...,None]
        replacements=dict(transfer=original+single,norm_matched=original+random,
                          paired=original+paired,whole_site=original[partner])
        injected=[single,random,paired]
        if borrowed is not None:
            old_donor,old_host,old_weights,old_bias=borrowed
            old_target=donor[partner]@old_donor@old_weights+old_bias
            direction=(old_target-original@old_host)@old_host.T
            magnitude=mx.linalg.norm(direction,axis=1,keepdims=True)
            assert bool(mx.all((magnitude>0)|(wanted==0)).item())
            edit=direction*(wanted/mx.maximum(magnitude,1e-12))
            replacements["borrowed"]=original+edit
            injected.append(edit)
        doses.extend(np.column_stack([np.array(mx.linalg.norm(z,axis=1)) for z in injected]))
        scores["native"].extend(whole_vocabulary(host,hidden,end,cut,choices))
        for name,value in replacements.items():
            modified=mx.where(place,value[:,None,:].astype(hidden.dtype),hidden)
            scores[name].extend(whole_vocabulary(host,modified,end,cut,choices))
    evidence={m:np.array(a) for m,a in scores.items()}
    doses=np.array(doses)
    np.testing.assert_allclose(doses[:,0],doses[:,1],atol=1e-6,rtol=1e-5)
    if borrowed is not None:
        np.testing.assert_allclose(doses[:,0],doses[:,-1],atol=1e-6,rtol=1e-5)
    assert all(a.shape==(len(rows),2,4) and np.isfinite(a).all() for a in evidence.values())
    return evidence,doses


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    paths={k:model_files(k) for k in SITES}
    dt,ht=[Tokenizer.from_file(str(paths[m]/"tokenizer.json")) for m in ("donor","host")]
    calibration=programs(ht,512,0,computed=False)
    assert calibration==programs(dt,512,0,computed=False)
    rows=programs(ht,512,114000,computed=True)
    source=programs(dt,512,115000,computed=False,values_by_pair=[r["source_values"] for r in rows])
    donor,_=load_model(paths["donor"],strict=True)
    donor_fit=gather(donor,calibration,SITES["donor"])
    states=gather(donor,source,SITES["donor"])
    del donor
    gc.collect()
    host,_=load_model(paths["host"],strict=True)
    host_fit=gather(host,calibration,SITES["host"])
    labels=[v for r in calibration for v in r["labels"]]
    axes=fit_map(donor_fit,host_fit,labels,classes=len(DIGITS))
    fit=fit_absolute(donor_fit,host_fit,axes)
    margins,norms=evaluate(host,rows,states,[mx.array(v) for v in axes],[mx.array(v) for v in fit])
    full=metrics(margins,full=True)
    forced=metrics(margins,full=False)
    decision="one_pass_transfer" if full["qualifies"] and full["reversals"]["paired"]>=.8*full["eligible"] else "unresolved"
    np.savez_compressed(out/"absolute.npz",norms=norms,**margins)
    np.savez_compressed(out/"map.npz",donor=axes[0],host=axes[1],orthogonal=axes[2],
                       paired_map=axes[3],absolute_map=fit[0],bias=fit[1])
    (out/"programs.json").write_text(json.dumps(dict(host=rows,donor=source),separators=(",",":"))+"\n")
    result=dict(decision=decision,models=MODELS,groups=len(rows),full_vocabulary=full,forced_two_choice=forced)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
