"""Test whether an identifier-selected attention route also carries a transplanted operand."""
from functools import partial
import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_attention_paths import read_masked
from identifier_attention_sparse import MODES,schedules
from identifier_baseline import SITES,encoded_batches,prefix
from identifier_binding import patch
from identifier_computation import programs
from identifier_numeric_open import vocabulary_ranks
from identifier_numeric_scale import PRIOR as MAP_JOB,archived_maps
from identifier_transfer import gather
from large_opener import MODELS,model_files

ROUTE_JOB="qwen-identifier-sparse-edge-sets"
ROUTE_REV="7f8d9bdd92d2e3ab0e918f213a0d29c79507f4f3"


def collect(host,rows,source_states,mapping,make_masks):
    donor_axis,host_axis,_,_,weights,bias=mapping
    cut=SITES["host"]
    scores={m+"_"+a:[] for m in MODES for a in ("clean","edited")}
    norms=[]
    offset=0
    for i,(ids,source,end,choices,_) in enumerate(encoded_batches(rows)):
        h=prefix(host,ids,cut)
        current=h[mx.arange(len(source)),source].astype(mx.float32)
        partner=mx.array(np.arange(len(source))^1)
        donor=mx.array(source_states[offset:offset+len(source)].astype(np.float32))
        offset+=len(source)
        target=donor[partner]@donor_axis@weights+bias
        change=(target-current@host_axis)@host_axis.T
        edited=patch(h,source,current+change)
        norms.extend(np.array(mx.linalg.norm(change,axis=1)))
        for m,mask in make_masks(ids.shape[1],source,end).items():
            scores[m+"_clean"].extend(read_masked(host,h,end,cut,choices,mask,scoring=vocabulary_ranks))
            scores[m+"_edited"].extend(read_masked(host,edited,end,cut,choices,mask,scoring=vocabulary_ranks))
        if (i+1)%8==0:
            print(f"Scored {min((i+1)*8,len(rows))}/{len(rows)} pairs",flush=True)
    return {k:np.array(a) for k,a in scores.items()},np.array(norms)


def summarize(scores):
    clean={m:(scores[m+"_clean"][:,:,0]==1).all(1) for m in MODES}
    common=np.logical_and.reduce(tuple(clean.values()))
    n=int(common.sum())
    report=dict(decision="unresolved",eligible=n,clean_groups={m:int(v.sum()) for m,v in clean.items()})
    if not n:
        return report
    flipped={m:(scores[m+"_edited"][common,:,1]==1).all(1) for m in MODES}
    counts={m:int(v.sum()) for m,v in flipped.items()}
    idx=np.random.default_rng(213816).integers(n,size=(5000,n))
    def contrast(a,b):
        return interval(flipped[a].astype(int)-flipped[b],idx)
    relative=contrast("strong_eight","unrestricted")
    high_low=contrast("strong_eight","weak_eight")
    removed=contrast("unrestricted","without_strong")
    valid=n>=100 and counts["unrestricted"]>=.8*n
    transfers=(valid and counts["blocked"]<=.15*n and counts["strong_eight"]>=.7*n
               and relative["interval"][0]>=-.15 and high_low["interval"][0]>.20 and removed["interval"][0]>.20)
    report.update(decision="route_transfers" if transfers else "unresolved",reversals=counts,
                  strong_minus_full=relative,strong_minus_weak=high_low,removal=removed,
                  unrestricted_control_qualified=valid)
    return report


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    route=root/ROUTE_JOB
    receipt=json.loads((route/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]==ROUTE_REV
    route_source=(route/"result.json").read_bytes()
    config=json.loads(route_source)
    assert config["decision"]=="sparse_attention_resolved"
    mapping,digest=archived_maps(root/MAP_JOB)
    donor_path,host_path=model_files("donor"),model_files("host")
    dt,ht=[Tokenizer.from_file(str(p/"tokenizer.json")) for p in (donor_path,host_path)]
    panel=programs(ht,512,119000,computed=True)
    source=programs(dt,512,120000,computed=False,values_by_pair=[r["source_values"] for r in panel])
    donor,_=load_model(donor_path,strict=True)
    states=gather(donor,source,SITES["donor"])
    del donor
    gc.collect()
    host,_=load_model(host_path,strict=True)
    assert len(host.model.layers)==48
    make=partial(schedules,cut=10,depth=48,strong=tuple(config["strong"]),weak=tuple(config["weak"]))
    scores,norms=collect(host,panel,states,mapping,make)
    assert all(v.shape==(512,2,4) and np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(out/"numeric-attention.npz",norms=norms,**scores)
    (out/"programs.json").write_text(json.dumps(dict(host=panel,donor=source),separators=(",",":"))+"\n")
    result=dict(models=MODELS,groups=len(panel),strong=config["strong"],weak=config["weak"],
                map_sha256=digest,route_sha256=sha256(route_source).hexdigest(),**summarize(scores))
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
