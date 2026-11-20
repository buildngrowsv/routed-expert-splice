"""Exchange a paired-letter subspace and a norm-matched random control."""
import gc
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, finish, make_pairs, prefix
from large_opener import MODELS, model_files


def bases(states, labels, classes=26):
    delta = states[::2]-states[1::2]
    onehot = np.eye(classes)[labels]
    y = onehot[::2]-onehot[1::2]
    gram = y.T@y
    fit = np.linalg.solve(gram+.02*np.trace(gram)/len(gram)*np.eye(len(gram)),y.T@delta)
    u,s,_ = np.linalg.svd(fit.T,full_matrices=False)
    rank = classes-1
    assert s[rank-1]>0
    left = u[:,:rank]
    random = np.random.default_rng(83407).normal(size=left.shape)
    random -= left@(left.T@random)
    control = np.linalg.qr(random,mode="reduced")[0]
    np.testing.assert_allclose(left.T@control,0,atol=1e-7)
    return left.astype(np.float32),control.astype(np.float32)


def norm_matched(diff, learned_basis, control_basis):
    learned = (diff@learned_basis)@learned_basis.T
    control = (diff@control_basis)@control_basis.T
    a = mx.linalg.norm(learned,axis=1,keepdims=True)
    b = mx.linalg.norm(control,axis=1,keepdims=True)
    assert bool(mx.all((b>0)|(a==0)).item())
    scaled = control*(a/mx.maximum(b,1e-12))
    np.testing.assert_allclose(np.array(a),np.array(mx.linalg.norm(scaled,axis=1,keepdims=True)),rtol=1e-5,atol=1e-6)
    return learned,scaled


def evaluate(model, calibration, test, depth, *, classes=26):
    x,y = [],[]
    for ids,site,_,_,labels in encoded_batches(calibration):
        state = prefix(model,ids,depth)
        x.append(np.array(state[mx.arange(len(site)),site].astype(mx.float32)))
        y.extend(labels)
    learned,control = bases(np.concatenate(x).astype(np.float64),np.asarray(y),classes)
    axes = [mx.array(a) for a in (learned,control)]
    modes = ("native","learned","matched","whole_site")
    scores = {k:[] for k in modes}
    dose = []
    for ids,site,end,candidates,_ in encoded_batches(test):
        h = prefix(model,ids,depth)
        src = h[mx.arange(len(site)),site]
        paired = src[mx.array(np.arange(len(site))^1)].astype(mx.float32)
        original = src.astype(mx.float32)
        delta = paired-original
        p,q = norm_matched(delta,*axes)
        dose.extend(np.column_stack([np.array(mx.linalg.norm(a,axis=1)) for a in (delta,p,q)]))
        at = (mx.arange(ids.shape[1])[None,:]==site[:,None])[...,None]
        scores["native"].extend(finish(model,h,end,depth,candidates))
        options = {"learned":original+p,"matched":original+q,"whole_site":paired}
        for name,value in options.items():
            altered = mx.where(at,value[:,None,:].astype(h.dtype),h)
            scores[name].extend(finish(model,altered,end,depth,candidates))
    margins = {k:np.array(v) for k,v in scores.items()}
    assert all(np.isfinite(v).all() for v in margins.values())
    hits = (margins["native"]>0).all(1)
    active = int(hits.sum())
    changes = {k:(v[hits]<0).all(1) for k,v in margins.items() if k!="native"}
    out = {k:int(v.sum()) for k,v in changes.items()}
    diff = changes["learned"].astype(int)-changes["matched"]
    if active:
        samples = np.random.default_rng(50324).integers(active,size=(5000,active))
        contrast = interval(diff,samples)
    else:
        contrast = {"mean":0.,"interval":[0.,0.]}
    pass_gate = (active>=100 and out["whole_site"]>=.2*active and out["learned"]>=.2*active
                 and contrast["mean"]>=.10 and contrast["interval"][0]>0)
    return margins,np.asarray(dose),dict(eligible=active,switches=out,learned_minus_matched=contrast,
                                         qualifies=bool(pass_gate),depth=depth,rank=classes-1)


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {k:model_files(k) for k in SITES}
    tokenizer = [Tokenizer.from_file(str(v/"tokenizer.json")) for v in paths.values()]
    train = make_pairs(tokenizer,512,0)
    test = make_pairs(tokenizer,512,8192)
    report = {}
    for key,cut in SITES.items():
        model,_ = load_model(paths[key],strict=True)
        margins,norms,report[key] = evaluate(model,train,test,cut)
        np.savez_compressed(out/(key+"-margins.npz"),norms=norms,**margins)
        del model
        gc.collect()
    (out/"test-programs.json").write_text(json.dumps(test,separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(dict(decision="complete",models=MODELS,calibration_pairs=len(train),
                                                  test_pairs=len(test),arms=report),indent=2)+"\n")


if __name__=="__main__":
    main()
