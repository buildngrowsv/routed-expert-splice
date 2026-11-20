"""Test whether access to an early key before the edit limits whole-state steering."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import encoded_batches
from identifier_binding import patch
from identifier_lookup_gap_72b import FRESH, REVISION
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import CUT, HOST
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files

ARMS = ("key_block", "syntax_block")


def attention(width, source, offset):
    position = mx.arange(width)
    query, key = position[None,:,None], position[None,None,:]
    source = source[:,None,None]
    return ((query>=key) & ~((query>source) & (key==source+offset)))[:,None,:,:]


def collect(model, rows):
    values = {m: {"native":[], "whole":[]} for m in ARMS}
    for i,(ids,site,end,choices,_) in enumerate(encoded_batches(rows),start=1):
        embedded = model.model.embed_tokens(ids)
        item = mx.arange(len(ids))
        for mode,offset in zip(ARMS,(0,-1),strict=True):
            mask = attention(ids.shape[1],site,offset)
            hidden = embedded
            for layer in model.model.layers[:CUT]:
                hidden=layer(hidden,mask)
            swapped=patch(hidden,site,hidden[item^1,site])
            values[mode]["native"].extend(whole_vocabulary(model,hidden,end,CUT,choices))
            values[mode]["whole"].extend(whole_vocabulary(model,swapped,end,CUT,choices))
        if i%12==0:
            print(f"Scored {i*8}/{len(rows)} contexts",flush=True)
    return {m:{k:np.array(x) for k,x in records.items()} for m,records in values.items()}


def score(arms):
    natives={m:good_groups(a["native"]) for m,a in arms.items()}
    match=np.logical_and.reduce(list(natives.values()))
    n=int(match.sum())
    switches, preserved={}, {}
    for mode,a in arms.items():
        z=a["whole"].reshape(-1,3,2,4)[match]
        switches[mode]=(z[:,:2,:,1]==1).all(axis=(1,2))
        preserved[mode]=int((z[:,2,:,0]==1).all(axis=1).sum())
    counts={m:int(x.sum()) for m,x in switches.items()}
    paired={}
    if n:
        sample=np.random.default_rng(877002).integers(n,size=(5000,n))
        for control in ("unrestricted","syntax_block"):
            paired[control]=interval(switches["key_block"].astype(int)-switches[control],sample)
    pass_gate=(n>=100 and counts["key_block"]>=.8*n and preserved["key_block"]>=.95*n
               and min(d["mean"] for d in paired.values())>=.15
               and min(d["interval"][0] for d in paired.values())>0)
    return dict(decision="blocking_preedit_key_access_restores_steering" if pass_gate else "unresolved",
                eligible=n,native={m:int(v.sum()) for m,v in natives.items()},
                switched=counts,constant_preserved=preserved,key_block_minus=paired)


def main():
    began=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/FRESH
    prior=verified_result(root,REVISION,"late_key_donor_transfer")
    raw=(root/"programs.json").read_bytes()
    assert sha256(raw).hexdigest()==prior["programs_digest"]
    rows=json.loads(raw)["early"]
    path=checkpoint_files(*HOST)
    tok=Tokenizer.from_file(str(path/"tokenizer.json"))
    for row in rows:
        site=row["site"]
        assert all(tok.decode([v[site-1]]).strip() in ("","=") for v in row["ids"])
    archive=root/"early.npz"
    with np.load(archive) as saved:
        unrestricted={k:saved[k] for k in ("native","whole")}
    model,_=load_model(path,strict=True)
    outcomes=collect(model,rows)
    outcomes["unrestricted"]=unrestricted
    result=score(outcomes)
    np.savez_compressed(out/"attention.npz",**{
        f"{m}_{k}":x for m,z in outcomes.items() if m in ARMS for k,x in z.items()
    })
    result.update(recipient=HOST,cut=CUT,source_programs_digest=prior["programs_digest"],
                  source_ranks_digest=sha256(archive.read_bytes()).hexdigest(),
                  runtime_s=time.perf_counter()-began,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
