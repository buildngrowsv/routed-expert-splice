"""Compare key exchange at its early declaration and at a late alias read."""
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
from identifier_baseline import encoded_batches,prefix
from identifier_binding import patch
from identifier_lookup_gap_72b import FRESH,REVISION
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import CUT,HOST
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files


def alias_rows(tokenizer,rows):
    output=[]
    for row in rows:
        prompts, ids, sites=[],[],[]
        for p, answer, answer_id in zip(row["prompts"],row["answers"],row["choices"],strict=True):
            lines=p.splitlines(keepends=True)
            key=lines[1].split("=")[0].strip()
            inserted=f"    selected = {key}\n"
            assert lines[3].startswith(f"    return table[{key}]")
            text="".join(lines[:3]+[inserted,lines[3].replace(f"table[{key}]", "table[selected]")]+lines[4:])
            encoded=tokenizer.encode(text,add_special_tokens=False)
            edge=sum(map(len,lines[:3]))+len(inserted.rstrip("\n"))
            reference_start=edge-len(key)
            members=[i for i,(lo,hi) in enumerate(encoded.offsets) if lo<edge and hi>reference_start]
            assert members and encoded.offsets[members[-1]][1]==edge
            assert tokenizer.encode(text+answer,add_special_tokens=False).ids==encoded.ids+[answer_id]
            exec(compile(text+answer+"\n","<archived explicit key copy>","exec"),{})
            prompts.append(text)
            ids.append(encoded.ids)
            sites.append(members[-1])
        changed=[i for i,(a,b) in enumerate(zip(*ids,strict=True)) if a!=b]
        assert changed==[row["site"]] and sites[0]==sites[1]>row["site"]
        output.append(dict(row,prompts=prompts,ids=ids,alias_site=sites[0]))
    return output


def evaluate(model,rows):
    scores={k:[] for k in ("native","early","alias")}
    for i,(ids,early,ends,digits,_) in enumerate(encoded_batches(rows)):
        targets=mx.array(np.repeat([r["alias_site"] for r in rows[i*8:i*8+len(ids)//2]],2))
        h=prefix(model,ids,CUT)
        index=mx.arange(len(ids))
        scores["native"].extend(whole_vocabulary(model,h,ends,CUT,digits))
        for label,site in (("early",early),("alias",targets)):
            patched=patch(h,site,h[index^1,site])
            scores[label].extend(whole_vocabulary(model,patched,ends,CUT,digits))
        if (i+1)%12==0:
            print(f"Scored {(i+1)*8}/{len(rows)} contexts",flush=True)
    return {k:np.array(v) for k,v in scores.items()}


def summarize(scores):
    live=good_groups(scores["native"])
    n=int(live.sum())
    flips,preserve={},{}
    for k in ("early","alias"):
        data=scores[k].reshape(-1,3,2,4)[live]
        flips[k]=(data[:,:2,:,1]==1).all(axis=(1,2))
        preserve[k]=int((data[:,2,:,0]==1).all(axis=1).sum())
    counts={k:int(v.sum()) for k,v in flips.items()}
    effect=None
    if n:
        draws=np.random.default_rng(951761).integers(n,size=(5000,n))
        effect=interval(flips["alias"].astype(int)-flips["early"],draws)
    success=(n>=100 and counts["alias"]>=.8*n and preserve["alias"]>=.95*n
             and effect["mean"]>=.15 and effect["interval"][0]>0)
    return dict(decision="late_alias_restores_whole_state_control" if success else "unresolved",
                eligible=n,switches=counts,preserved=preserve,alias_minus_early=effect)


def main():
    start=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    source=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/FRESH
    prior=verified_result(source,REVISION,"late_key_donor_transfer")
    raw=(source/"programs.json").read_bytes()
    signature=sha256(raw).hexdigest()
    assert signature==prior["programs_digest"]
    path=checkpoint_files(*HOST)
    tok=Tokenizer.from_file(str(path/"tokenizer.json"))
    rows=alias_rows(tok,json.loads(raw)["early"])
    model,_=load_model(path,strict=True)
    scores=evaluate(model,rows)
    np.savez_compressed(out/"alias.npz",**scores)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    report=summarize(scores)
    report.update(recipient=HOST,cut=CUT,groups=len(rows)//3,source_programs_digest=signature,
                  runtime_s=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
