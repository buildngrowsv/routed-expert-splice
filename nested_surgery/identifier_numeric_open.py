"""Remove the forced answer shortlist when scoring the pinned operand transfer."""

import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np

from bridge_input import interval
from identifier_baseline import SITES
from identifier_numeric_composition import MAP_JOB, load_prior
from identifier_transfer import collect_edits, gather
from large_opener import MODELS, model_files


def whole_vocabulary(model,h,ends,cut,candidates):
    mask = create_attention_mask(h,None)
    for layer in model.model.layers[cut:]:
        h = layer(h,mask)
    item = mx.arange(len(ends))
    vectors = model.model.norm(h[item,ends])
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    z = head(vectors).astype(mx.float32)
    true_idx = mx.array(candidates)
    other_idx = mx.array(candidates[np.arange(len(candidates))^1])
    true_logit = z[item,true_idx]
    other_logit = z[item,other_idx]
    true_rank = mx.sum(z>=true_logit[:,None],axis=1)
    other_rank = mx.sum(z>=other_logit[:,None],axis=1)
    top = mx.argmax(z,axis=1)
    return np.asarray(mx.stack((true_rank,other_rank,top,true_logit-other_logit),axis=1)).reshape(-1,2,4)


def metrics(data, *, full):
    base = data["native"]
    clean = ((base[:,:,0]==1) if full else (base[:,:,3]>0)).all(1)
    n = int(clean.sum())
    turned = {k:((v[clean,:,1]==1) if full else (v[clean,:,3]<0)).all(1)
              for k,v in data.items() if k!="native"}
    counts = {k:int(v.sum()) for k,v in turned.items()}
    effect = None
    if n:
        diff = turned["transfer"].astype(int)-turned["norm_matched"]
        idx = np.random.default_rng(824625).integers(n,size=(4000,n))
        effect = interval(diff,idx)
    passed = (n>=100 and min(counts["whole_site"],counts["transfer"])>=.8*n
              and effect["mean"]>=.2 and effect["interval"][0]>0)
    return dict(eligible=n,reversals=counts,transfer_minus_random=effect,qualifies=bool(passed))


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    prior = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/MAP_JOB
    axes,digest = load_prior(prior)
    archive = (prior/"programs.json").read_bytes()
    rows = json.loads(archive)
    donor,_ = load_model(model_files("donor"),strict=True)
    state = gather(donor,rows["donor"],SITES["donor"])
    del donor
    gc.collect()
    host,_ = load_model(model_files("host"),strict=True)
    scores,norms = collect_edits(host,rows["host"],state,axes,whole_vocabulary)
    assert all(v.shape==(512,2,4) for v in scores.values())
    full = metrics(scores,full=True)
    forced = metrics(scores,full=False)
    np.savez_compressed(out/"open-vocabulary.npz",norms=norms,**scores)
    result = dict(decision="unrestricted_transfer" if full["qualifies"] else "unresolved",
                  models=MODELS,groups=len(rows["host"]),map_sha256=digest,source_sha256=sha256(archive).hexdigest(),
                  full_vocabulary=full,forced_two_choice=forced)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
