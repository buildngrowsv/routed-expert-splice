"""Hold a trained letter subspace constant across seven frozen transformer blocks."""
import gc
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, finish, make_pairs, prefix
from identifier_subspace import bases, norm_matched
from large_opener import MODELS, model_files


def constrained(model,h,site,end,depth,candidates,axis,target):
    mask = create_attention_mask(h,None)
    token = mx.arange(len(site))
    place = (mx.arange(h.shape[1])[None,:]==site[:,None])[...,None]
    for block in model.model.layers[depth:depth+7]:
        value = h[token,site].astype(mx.float32)
        correction = (target-value@axis)@axis.T
        updated = (value+correction).astype(h.dtype)
        h = mx.where(place,updated[:,None,:],h)
        h = block(h,mask)
    value = h[token,site].astype(mx.float32)
    corrected = (value+(target-value@axis)@axis.T).astype(h.dtype)
    return finish(model,mx.where(place,corrected[:,None,:],h),end,depth+7,candidates)


def evaluate(model,cal,test,depth):
    x,y = [],[]
    for ids,site,_,_,labels in encoded_batches(cal):
        h = prefix(model,ids,depth)
        x.append(np.array(h[mx.arange(len(site)),site].astype(mx.float32)))
        y.extend(labels)
    learned,matched = map(mx.array,bases(np.concatenate(x).astype(np.float64),np.asarray(y)))
    scores = {k:[] for k in ("native","learned_clean","learned_swap","random_clean","random_swap")}
    for ids,site,end,choices,_ in encoded_batches(test):
        h = prefix(model,ids,depth)
        v = h[mx.arange(len(site)),site].astype(mx.float32)
        partner = v[mx.array(np.arange(len(site))^1)]
        _,control = norm_matched(partner-v,learned,matched)
        scores["native"].extend(finish(model,h,end,depth,choices))
        modes = (("learned",learned,v,partner),("random",matched,v,v+control))
        for k,b,clean,swapped in modes:
            for side,source in (("clean",clean),("swap",swapped)):
                z = constrained(model,h,site,end,depth,choices,b,source@b)
                scores[k+"_"+side].extend(z)
    values = {k:np.array(v) for k,v in scores.items()}
    assert all(np.isfinite(v).all() for v in values.values())
    native = (values["native"]>0).all(1)
    lc = native & (values["learned_clean"]>0).all(1)
    rc = native & (values["random_clean"]>0).all(1)
    overlap = lc & rc
    lr,rr = [(values[k]<0).all(1) for k in ("learned_swap","random_swap")]
    n,l,common = map(int,(native.sum(),lc.sum(),overlap.sum()))
    reversals = int((lc&lr).sum())
    contrast = dict(mean=0.,interval=[0.,0.])
    if common:
        rng = np.random.default_rng(60217)
        draw = rng.integers(common,size=(4000,common))
        contrast = interval(lr[overlap].astype(int)-rr[overlap],draw)
    qualifies = n>=100 and l>=.9*n and reversals>=.8*l and common>=100
    qualifies = qualifies and contrast["mean"]>=.2 and contrast["interval"][0]>0
    record = dict(native=n,learned_retained=l,random_retained=int(rc.sum()),matched_eligible=common,
                  learned_reversed=reversals,learned_reversed_on_matched=int((lr&overlap).sum()),
                  random_reversed_on_matched=int((rr&overlap).sum()),contrast=contrast,qualifies=bool(qualifies))
    return values,record


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {k:model_files(k) for k in SITES}
    tok = [Tokenizer.from_file(str(v/"tokenizer.json")) for v in paths.values()]
    cal,test = make_pairs(tok,512,0),make_pairs(tok,512,12000)
    report = {}
    for k,cut in SITES.items():
        model,_ = load_model(paths[k],strict=True)
        data,report[k] = evaluate(model,cal,test,cut)
        np.savez_compressed(out/(k+"-margins.npz"),**data)
        del model
        gc.collect()
    (out/"test-programs.json").write_text(json.dumps(test,separators=(",",":"))+"\n")
    summary = dict(decision="complete",models=MODELS,groups=len(test),reference_blocks=7,arms=report)
    (out/"result.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
