"""Separate the first hop of a transplanted arithmetic operand by token span."""
from functools import partial
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import re

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES
from identifier_computation import programs
from identifier_numeric_attention import collect
from identifier_numeric_scale import PRIOR as MAP_JOB,archived_maps
from identifier_transfer import gather
from large_opener import MODELS,model_files

MODES=("unrestricted","all_relays","use_only","control_only","without_use")
PRIOR="qwen-numeric-identifier-route-transfer"


def paths_for_programs(tokenizer,rows):
    bounds=[]
    for row in rows:
        pair=[]
        for text,ids in zip(row["prompts"],row["ids"],strict=True):
            match=re.search(r"return (amount_\d+)",text)
            assert match
            a,b=match.span(1)
            encoded=tokenizer.encode(text,add_special_tokens=False)
            assert encoded.ids==ids
            between=[j for j,(start,end) in enumerate(encoded.offsets) if end>a and start<b]
            first,last=min(between),max(between)+1
            assert between==list(range(first,last)) and row["site"]<first
            comparison=first-(last-first)
            assert row["site"]<comparison and last<=len(ids)-1
            pair.append((first,last,comparison,first))
        assert pair[0]==pair[1]
        row["relay_spans"]=dict(use=list(pair[0][:2]),control=list(pair[0][2:]))
        bounds.extend(pair)
    return mx.array(bounds)


def make_paths(width,source,end,start,*,bounds):
    p=mx.arange(width)
    q=p[None,:,None]
    k=p[None,None,:]
    causal=q>=k
    source_key=k==source[:,None,None]
    self_read=q==source[:,None,None]
    final=q==end[:,None,None]
    selected=bounds[start:start+len(source)]
    use_begin,use_end,control_begin,control_end=[selected[:,i,None,None] for i in range(4)]
    use=(q>=use_begin)&(q<use_end)
    control=(q>=control_begin)&(q<control_end)
    unrestricted=mx.broadcast_to(causal,(len(source),width,width))
    relays=causal&(~source_key|~final)
    allowed_use=causal&(~source_key|self_read|use)
    allowed_control=causal&(~source_key|self_read|control)
    remove_use=causal&(~source_key|~(use|final))
    return {name:mask[:,None] for name,mask in zip(
        MODES,(unrestricted,relays,allowed_use,allowed_control,remove_use),strict=True)}


def summarize(scores):
    clean={m:(scores[m+"_clean"][:,:,0]==1).all(1) for m in MODES}
    keep=np.logical_and.reduce(tuple(clean.values()))
    n=int(keep.sum())
    summary=dict(decision="unresolved",eligible=n,native_by_mask={m:int(v.sum()) for m,v in clean.items()})
    if not n:
        return summary
    switched={m:(scores[m+"_edited"][keep,:,1]==1).all(1) for m in MODES}
    counts={m:int(v.sum()) for m,v in switched.items()}
    draws=np.random.default_rng(824939).integers(n,size=(5000,n))
    def compare(a,b):
        return interval(switched[a].astype(int)-switched[b],draws)
    near=compare("use_only","all_relays")
    matched=compare("use_only","control_only")
    remove=compare("all_relays","without_use")
    valid=n>=100 and counts["unrestricted"]>=.8*n and counts["all_relays"]>=.6*n
    sufficient=valid and counts["use_only"]>=.6*n and near["interval"][0]>=-.15
    leads=valid and matched["interval"][0]>.20
    necessary=valid and remove["interval"][0]>.20
    summary.update(decision="relay_use_supported" if sufficient or leads or necessary else "unresolved",
                   reversals=counts,relative_to_relays=near,use_minus_control=matched,removed_use=remove,
                   sufficient=sufficient,use_leads=leads,removal_material=necessary,controls_qualified=valid)
    return summary


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    previous=(root/PRIOR/"result.json").read_bytes()
    info=json.loads(previous)
    assert info["reversals"]["blocked"]>=.6*info["eligible"]
    mapping,digest=archived_maps(root/MAP_JOB)
    donor_path,host_path=model_files("donor"),model_files("host")
    donor_tok,host_tok=[Tokenizer.from_file(str(p/"tokenizer.json")) for p in (donor_path,host_path)]
    panel=programs(host_tok,512,121000,computed=True)
    bounds=paths_for_programs(host_tok,panel)
    source=programs(donor_tok,512,122000,computed=False,values_by_pair=[r["source_values"] for r in panel])
    donor,_=load_model(donor_path,strict=True)
    features=gather(donor,source,SITES["donor"])
    del donor
    gc.collect()
    host,_=load_model(host_path,strict=True)
    scores,norm=collect(host,panel,features,mapping,partial(make_paths,bounds=bounds),modes=MODES)
    assert all(v.shape==(512,2,4) and np.isfinite(v).all() for v in scores.values())
    np.savez_compressed(out/"relay.npz",norms=norm,**scores)
    (out/"programs.json").write_text(json.dumps(dict(host=panel,donor=source),separators=(",",":"))+"\n")
    report=dict(models=MODELS,groups=len(panel),earlier_route_sha256=sha256(previous).hexdigest(),
                map_sha256=digest,**summarize(scores))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
