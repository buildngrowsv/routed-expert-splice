"""Test whether an alias becomes sufficient at its actual dictionary-use site."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import encoded_batches, prefix
from identifier_binding import patch
from identifier_numeric_carry_72b import HOST
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files

HERE=Path(__file__).resolve().parent
CUT=40
ARMS=("lookup_read","close_bracket","joint")


def positions(tokenizer, rows):
    sites=[]
    for row in rows:
        pair=[]
        for prompt, ids in zip(row["prompts"],row["ids"],strict=True):
            first=prompt.index("table[selected]")+len("table[")
            last=first+len("selected")
            encoded=tokenizer.encode(prompt,add_special_tokens=False)
            assert encoded.ids==ids
            read,=[i for i,(a,b) in enumerate(encoded.offsets) if a<last and b>first]
            close,=[i for i,(a,b) in enumerate(encoded.offsets) if a<=last<b]
            assert close==read+1 and read>row["alias_site"]
            pair.append((read,close))
        assert pair[0]==pair[1]
        sites.append(pair[0])
    return np.asarray(sites,dtype=np.int32)


def execute(model, rows, loc, reference):
    data={k:[] for k in ARMS}
    for i,(ids,_,ends,candidates,_) in enumerate(encoded_batches(rows)):
        h=prefix(model,ids,CUT)
        if i==0:
            before=whole_vocabulary(model,h,ends,CUT,candidates)
            np.testing.assert_array_equal(before[...,:3],reference[:len(ids)//2,:,:3])
        index=mx.arange(len(ids))^1
        read,close=(mx.array(a) for a in np.repeat(loc[i*8:i*8+len(ids)//2],2,axis=0).T)
        first=patch(h,read,h[index,read])
        last=patch(h,close,h[index,close])
        both=patch(first,close,h[index,close])
        for key,value in zip(ARMS,(first,last,both),strict=True):
            data[key].append(whole_vocabulary(model,value,ends,CUT,candidates))
        if (i+1)%12==0:
            print(f"Scored {min((i+1)*8,len(rows))}/{len(rows)} dictionary contexts",flush=True)
    return {k:np.concatenate(v) for k,v in data.items()}


def summarize(scores, native, control, old):
    eligible=good_groups(native)&good_groups(control)
    n=int(eligible.sum())
    assert n>=100
    counts, switched={},{}
    for label,array in (old|scores).items():
        retained=array.reshape(-1,3,2,4)[eligible]
        switched[label]=(retained[:,:2,:,1]==1).all(axis=(1,2))
        counts[label]={"switch":int(switched[label].sum()),
                       "preserve":int((retained[:,2,:,0]==1).all(axis=1).sum())}
    draws=np.random.default_rng(741195).integers(n,size=(8000,n))
    comparisons={}
    for arm in ARMS:
        rivals=("alias_copy",) if arm!="joint" else ("alias_copy","lookup_read","close_bracket")
        comparisons[arm]={}
        for against in rivals:
            difference=switched[arm].astype(int)-switched[against]
            comparisons[arm][against]=dict(mean=float(difference.mean()),
                                         interval=np.quantile(difference[draws].mean(1),[.0125,.9875]).tolist())
    def useful(k):
        delta=comparisons[k]["alias_copy"]
        return (counts[k]["switch"]>=.5*n and counts[k]["preserve"]>=.95*n
                and delta["mean"]>=.2 and delta["interval"][0]>0)
    complements=(useful("joint") and all(comparisons["joint"][k]["mean"]>=.10 and comparisons["joint"][k]["interval"][0]>0
                                         for k in ("lookup_read","close_bracket")))
    singles=[k for k in ("lookup_read","close_bracket") if useful(k)]
    decision="unresolved"
    if complements:
        decision="joint_alias_lookup_span_steers"
    elif singles:
        decision="alias_use_site_steers"
    return dict(decision=decision,shared=n,counts=counts,paired=comparisons,qualified_single_sites=singles)


def main():
    start=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    initial=HERE/"numeric_lookup_alias_72b_study/results"
    depths=HERE/"numeric_lookup_alias_depth_72b_study/results"
    result=json.loads((depths/"result.json").read_text())
    assert result["decision"]=="alias_gap_persists_at_viable_cuts" and 40 in result["viable_cuts"]
    raw=(initial/"programs.json").read_bytes()
    assert sha256(raw).hexdigest()==result["alias_programs_sha256"]
    rows=json.loads(raw)
    host=checkpoint_files(*HOST)
    tokenizer=Tokenizer.from_file(str(host/"tokenizer.json"))
    loc=positions(tokenizer,rows)
    model,_=load_model(host,strict=True)
    with np.load(depths/"depth.npz") as z:
        n,control=z["native_alias"],z["native_relocated"]
        older={"alias_copy":z["alias_40"],"relocated_definition":z["relocated_40"]}
    scores=execute(model,rows,loc,n)
    report=summarize(scores,n,control,older)
    assert report["shared"]==result["shared"] and report["counts"]["alias_copy"]==result["metrics"]["alias_40"]
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"alias-use.npz",**scores,locations=loc)
    report.update(cut=CUT,model=HOST,panel_sha256=sha256(raw).hexdigest(),
                  compared_ranks_sha256=sha256((depths/"depth.npz").read_bytes()).hexdigest(),
                  seconds=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
