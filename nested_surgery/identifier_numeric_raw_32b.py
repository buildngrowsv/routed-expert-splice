"""Use the 14B interface in 32B without obtaining its edit norm from the recipient."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np

from bridge_input import interval
from identifier_baseline import SITES,encoded_batches,prefix
from identifier_binding import patch
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_scale import HOST_32,PRIOR as MAP_JOB,archived_maps
from identifier_transfer import gather
from large_opener import MODELS,checkpoint_files,model_files

PRIOR="qwen-one-pass-affine-32b"
REVISION="ea4014f0edf250fa8abf5d0bb52657371998aea3"
MODES=("native","borrowed_raw","norm_matched")


def evaluate(host,rows,states,basis,*,cut=13,gain=1.0):
    donor,receiver,random,_,weights,bias=basis
    result={k:[] for k in MODES}
    norms=[]
    offset=0
    for step,(ids,site,end,choices,_) in enumerate(encoded_batches(rows)):
        h=prefix(host,ids,cut)
        source=h[mx.arange(len(site)),site].astype(mx.float32)
        x=mx.array(states[offset:offset+len(site)].astype(np.float32))
        offset+=len(site)
        target=x[mx.array(np.arange(len(site))^1)]@donor@weights+bias
        delta=target-source@receiver
        raw=gain*(delta@receiver.T)
        control=delta@random.T
        a=mx.linalg.norm(raw,axis=1,keepdims=True)
        b=mx.linalg.norm(control,axis=1,keepdims=True)
        assert bool(mx.all((b>0)|(a==0)).item())
        control=control*(a/mx.maximum(b,1e-12))
        norms.extend(np.column_stack((np.array(a).ravel(),np.array(mx.linalg.norm(control,axis=1)))))
        result["native"].extend(whole_vocabulary(host,h,end,cut,choices))
        for mode,d in (("borrowed_raw",raw),("norm_matched",control)):
            result[mode].extend(whole_vocabulary(host,patch(h,site,source+d),end,cut,choices))
        if (step+1)%8==0:
            print(f"Scored {min((step+1)*8,len(rows))}/{len(rows)} pairs",flush=True)
    shapes={k:np.array(v) for k,v in result.items()}
    norms=np.array(norms)
    assert all(a.shape==(len(rows),2,4) and np.isfinite(a).all() for a in shapes.values())
    np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
    return shapes,norms


def summarize(scores,prior):
    native=(scores["native"][:,:,0]==1).all(1)
    archived=(prior["native"][:,:,0]==1).all(1)
    eligible=native&archived
    n=int(eligible.sum())
    outcomes={m:(v[eligible,:,1]==1).all(1) for m,v in scores.items() if m!="native"}
    outcomes["borrowed_scaled"]=(prior["borrowed"][eligible,:,1]==1).all(1)
    counts={m:int(v.sum()) for m,v in outcomes.items()}
    raw_random=raw_scaled=None
    if n:
        idx=np.random.default_rng(640393).integers(n,size=(5000,n))
        raw=outcomes["borrowed_raw"].astype(int)
        raw_random=interval(raw-outcomes["norm_matched"],idx)
        raw_scaled=interval(raw-outcomes["borrowed_scaled"],idx)
    passed=(n>=100 and counts["borrowed_raw"]>=.75*n and raw_random["mean"]>=.20 and raw_random["interval"][0]>0)
    return dict(decision="unscaled_transfer" if passed else "unresolved",eligible=n,
                native_new=int(native.sum()),native_prior=int(archived.sum()),reversals=counts,
                raw_minus_random=raw_random,raw_minus_scaled=raw_scaled,
                near_scaled=bool(raw_scaled and raw_scaled["interval"][0]>=-.10))


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    prior=root/PRIOR
    receipt=json.loads((prior/"receipt.json").read_text())
    assert receipt["revision"]==REVISION and receipt["status"]=="done"
    assert json.loads((prior/"result.json").read_text())["decision"]=="scales_to_32b"
    stored=(prior/"programs.json").read_bytes()
    data=json.loads(stored)
    donor,_=load_model(model_files("donor"),strict=True)
    states=gather(donor,data["donor"],SITES["donor"])
    del donor
    gc.collect()
    directions,digest=archived_maps(root/MAP_JOB)
    host,_=load_model(checkpoint_files(*HOST_32),strict=True)
    assert len(host.model.layers)==64
    scores,norms=evaluate(host,data["host"],states,directions)
    with np.load(prior/"32b.npz") as z:
        result=summarize(scores,z)
        old_norms=z["norms"][:,3]
    assert (old_norms>0).all()
    relative=np.quantile(norms[:,0]/old_norms,[.1,.5,.9]).tolist()
    np.savez_compressed(out/"raw-32b.npz",norms=norms,**scores)
    report=dict(models={"donor":MODELS["donor"],"host":HOST_32},groups=len(data["host"]),
                map_digest=digest,prompts_digest=sha256(stored).hexdigest(),norm_ratio_quantiles=relative,**result)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
