"""Translate a 25-dimensional causal change from the 7B donor into the 14B host."""
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, finish, make_pairs, prefix
from identifier_subspace import bases
from large_opener import MODELS, model_files


def gather(model, rows, cut):
    out = []
    for ids,site,_,_,_ in encoded_batches(rows):
        h = prefix(model,ids,cut)
        out.append(np.array(h[mx.arange(len(site)),site].astype(mx.float32)))
    return np.concatenate(out).astype(np.float64)


def fit_map(donor, host, labels, *, classes=26):
    donor_basis,_ = bases(donor,labels,classes)
    host_basis,random_basis = bases(host,labels,classes)
    x = (donor[1::2]-donor[::2])@donor_basis
    y = (host[1::2]-host[::2])@host_basis
    gram = x.T@x
    penalty = .02*np.trace(gram)/len(gram)
    mapping = np.linalg.solve(gram+penalty*np.eye(len(gram)),x.T@y)
    return donor_basis,host_basis,random_basis,mapping.astype(np.float32)


def transform(donor, src_axis, dst_axis, control_axis, mapping):
    donor_delta = donor[mx.array(np.arange(len(donor))^1)]-donor
    coordinates = (donor_delta@src_axis)@mapping
    learned = coordinates@dst_axis.T
    control = coordinates@control_axis.T
    left = mx.linalg.norm(learned,axis=1,keepdims=True)
    right = mx.linalg.norm(control,axis=1,keepdims=True)
    assert bool(mx.all((right>0)|(left==0)).item())
    control = control*(left/mx.maximum(right,1e-12))
    np.testing.assert_allclose(np.array(left),np.array(mx.linalg.norm(control,axis=1,keepdims=True)),
                               atol=1e-6,rtol=1e-5)
    return learned,control


def collect_edits(host, rows, donor_states, axes, readout):
    result = {k:[] for k in ("native","transfer","norm_matched","whole_site")}
    doses = []
    for i,(ids,site,end,choices,_) in enumerate(encoded_batches(rows)):
        cut = SITES["host"]
        h = prefix(host,ids,cut)
        place = (mx.arange(ids.shape[1])[None,:]==site[:,None])[...,None]
        source = h[mx.arange(len(site)),site].astype(mx.float32)
        start = i*16
        learned,random = transform(mx.array(donor_states[start:start+len(site)].astype(np.float32)),*axes)
        # The host's partner state is used only in the positive control.
        full = source[mx.array(np.arange(len(site))^1)]
        replacements = {"transfer":source+learned,"norm_matched":source+random,"whole_site":full}
        doses.extend(np.column_stack([np.array(mx.linalg.norm(a,axis=1)) for a in (learned,random,full-source)]))
        result["native"].extend(readout(host,h,end,cut,choices))
        for k,value in replacements.items():
            state = mx.where(place,value[:,None,:].astype(h.dtype),h)
            result[k].extend(readout(host,state,end,cut,choices))
    scores = {k:np.asarray(v) for k,v in result.items()}
    assert all(np.isfinite(v).all() for v in scores.values())
    return scores,np.asarray(doses)


def evaluate(host, rows, donor_states, axes):
    scores,doses = collect_edits(host,rows,donor_states,axes,finish)
    active = (scores["native"]>0).all(1)
    n = int(active.sum())
    switched = {k:((v<0).all(1)&active)[active] for k,v in scores.items() if k!="native"}
    counts = {k:int(v.sum()) for k,v in switched.items()}
    diff = switched["transfer"].astype(int)-switched["norm_matched"]
    comparison = interval(diff,np.random.default_rng(720504).integers(n,size=(4000,n))) if n else None
    qualifies = (n>=100 and counts["whole_site"]>=.8*n and counts["transfer"]>=.8*n
                 and comparison["mean"]>=.2 and comparison["interval"][0]>0)
    report = dict(decision="transfer" if qualifies else "stop",eligible=n,switches=counts,
                  item_switches={k:((v[active]<0).sum(0)).tolist() for k,v in scores.items() if k!="native"},
                  transfer_minus_random=comparison)
    return scores,doses,report


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {k:model_files(k) for k in SITES}
    tok = [Tokenizer.from_file(str(v/"tokenizer.json")) for v in paths.values()]
    cal,test = make_pairs(tok,512,0),make_pairs(tok,512,16000)
    donor,_ = load_model(paths["donor"],strict=True)
    source = gather(donor,cal,SITES["donor"])
    donor_test = gather(donor,test,SITES["donor"])
    del donor
    host,_ = load_model(paths["host"],strict=True)
    target = gather(host,cal,SITES["host"])
    labels = [ord(c)-97 for row in cal for c in row["letters"]]
    matrices = fit_map(source,target,labels)
    scores,dose,summary = evaluate(host,test,donor_test,[mx.array(m) for m in matrices])
    np.savez_compressed(out/"transfer.npz",norms=dose,**scores)
    np.savez_compressed(out/"map.npz",**dict(zip(("donor_axis","host_axis","random_host_axis","mapping"),matrices)))
    (out/"test-programs.json").write_text(json.dumps(test,separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(dict(models=MODELS,groups=len(test),calibration_pairs=len(cal),rank=25,
                                                  **summary),indent=2)+"\n")


if __name__=="__main__":
    main()
