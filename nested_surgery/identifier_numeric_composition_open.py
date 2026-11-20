"""Score two independent donor operands without a response shortlist."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np

from bridge_input import interval
from identifier_baseline import SITES
from identifier_numeric_composition import ARMS, MAP_JOB, completion_logits, evaluate, load_prior
from identifier_transfer import gather, transform
from large_opener import MODELS, model_files

MODES=("native","both","random","whole")
PRIOR="qwen-two-operand-native-qualification"
REVISION="39538af4cfef39c536bb695d56c8a3ec216927b6"


def rank_readout(model,h,ends,cut,choices):
    z=completion_logits(model,h,ends,cut)
    shortlisted=z[mx.arange(len(ends))[:,None],choices]
    ranks=mx.sum(z[:,None,:]>=shortlisted[:,:,None],axis=2)
    values=mx.concatenate((shortlisted,ranks,mx.argmax(z,axis=1)[:,None]),axis=1)
    return np.asarray(values).reshape(-1,4,9)


def summarize(scores,panel,*,full):
    outcomes={}
    labels=np.arange(4)
    for mode,v in scores.items():
        target=(labels^ARMS[mode])[None,:,None]
        if full:
            won=np.take_along_axis(v[:,:,4:8],target,axis=2)[...,0]==1
        else:
            restricted=v[:,:,:4]
            correct=np.take_along_axis(restricted,target,axis=2)[...,0]
            other=np.where(labels[None,None,:]==target,-np.inf,restricted).max(2)
            won=correct>other
        outcomes[mode]=won.all(1)
    eligible=outcomes["native"]
    n=int(eligible.sum())
    counts={k:int(v[eligible].sum()) for k,v in outcomes.items()}
    sign=np.array([r["sign"] for r in panel])
    by_operator={}
    for name,mask in (("addition",sign>0),("subtraction",sign<0)):
        subset=eligible&mask
        by_operator[name]=dict(eligible=int(subset.sum()),correct={k:int(v[subset].sum()) for k,v in outcomes.items()})
    effect=None
    if n:
        idx=np.random.default_rng(214790).integers(n,size=(5000,n))
        effect=interval(outcomes["both"][eligible].astype(int)-outcomes["random"][eligible],idx)
    qualified=(n>=100 and min(counts["both"],counts["whole"])>=.8*n
               and effect["mean"]>=.2 and effect["interval"][0]>0)
    return dict(eligible=n,correct=counts,both_minus_random=effect,qualifies=qualified,by_operator=by_operator)


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    prior=root/PRIOR
    receipt=json.loads((prior/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]==REVISION
    reference=json.loads((prior/"result.json").read_text())
    assert reference["decision"]=="composes"
    axes,digest=load_prior(root/MAP_JOB)
    assert digest==reference["map_digest"]
    raw=(prior/"programs.json").read_bytes()
    rows=json.loads(raw)
    panel=rows["host"]
    donor,_=load_model(model_files("donor"),strict=True)
    changes,norms={},{}
    for side,source in rows["donor"].items():
        states=gather(donor,source,SITES["donor"]).astype(np.float32)
        d=transform(mx.array(states),*axes)
        changes[side]=[np.array(a).reshape(len(panel),2,-1) for a in d]
        norms[side]=np.stack([np.linalg.norm(a,axis=2) for a in changes[side]],axis=-1)
    del donor
    gc.collect()
    host,_=load_model(model_files("host"),strict=True)
    scores=evaluate(host,panel,changes,read=rank_readout,arms=MODES)
    assert all(v.shape==(512,4,9) and np.isfinite(v).all() for v in scores.values())
    full=summarize(scores,panel,full=True)
    forced=summarize(scores,panel,full=False)
    result=dict(decision="composition_unrestricted" if full["qualifies"] else "unresolved",
                models=MODELS,groups=512,map_digest=digest,source_digest=sha256(raw).hexdigest(),
                full_vocabulary=full,forced_four_choice=forced)
    np.savez_compressed(out/"composition-open.npz",left_norms=norms["left"],right_norms=norms["right"],**scores)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
