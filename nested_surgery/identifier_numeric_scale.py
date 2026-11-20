"""Compare original and receiver-fitted one-pass interfaces at 14B and 32B."""
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
from identifier_baseline import SITES
from identifier_computation import DIGITS,programs
from identifier_numeric_absolute import evaluate,fit_absolute
from identifier_numeric_open import metrics
from identifier_transfer import fit_map,gather
from large_opener import MODELS,checkpoint_files,model_files

HOST_32=("Qwen/Qwen2.5-32B","1818d35814b8319459f4bd55ed1ac8709630f003")
PRIOR="qwen-numeric-one-pass-affine"
REVISION="e29adff07c4d132863e855b93d1319ed5ebed808"


def archived_maps(job):
    receipt=json.loads((job/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]==REVISION
    assert json.loads((job/"result.json").read_text())["decision"]=="one_pass_transfer"
    path=job/"map.npz"
    with np.load(path) as data:
        names=("donor","host","orthogonal","paired_map","absolute_map","bias")
        values=tuple(mx.array(data[k]) for k in names)
    return values,sha256(path.read_bytes()).hexdigest()


def shared_scores(s14,s32):
    agreed=(s14["native"][:,:,0]==1).all(1)&(s32["native"][:,:,0]==1).all(1)
    n=int(agreed.sum())
    result=dict(eligible=n,arms={})
    for name,scores in (("14b",s14),("32b",s32)):
        result["arms"][name]={k:int((v[agreed,:,1]==1).all(1).sum()) for k,v in scores.items() if k!="native"}
    return result


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    queue=Path(os.environ["STUDIO_QUEUE_ROOT"])
    p32=checkpoint_files(*HOST_32)
    paths={"donor":model_files("donor"),"14b":model_files("host"),"32b":p32}
    tok={k:Tokenizer.from_file(str(p/"tokenizer.json")) for k,p in paths.items()}
    cal=programs(tok["14b"],512,0,computed=False)
    assert cal==programs(tok["32b"],512,0,computed=False)==programs(tok["donor"],512,0,computed=False)
    panel=programs(tok["14b"],512,117000,computed=True)
    assert panel==programs(tok["32b"],512,117000,computed=True)
    source=programs(tok["donor"],512,118000,computed=False,values_by_pair=[r["source_values"] for r in panel])
    donor,_=load_model(paths["donor"],strict=True)
    cal_donor=gather(donor,cal,SITES["donor"])
    target_donor=gather(donor,source,SITES["donor"])
    del donor
    gc.collect()
    old,digest=archived_maps(queue/"runs"/PRIOR)
    host,_=load_model(paths["14b"],strict=True)
    assert len(host.model.layers)==48
    s14,n14=evaluate(host,panel,target_donor,old[:4],old[4:],cut=10)
    del host
    gc.collect()
    host,_=load_model(paths["32b"],strict=True)
    assert len(host.model.layers)==64
    cal_host=gather(host,cal,13)
    labels=[v for row in cal for v in row["labels"]]
    fresh=fit_map(cal_donor,cal_host,labels,classes=len(DIGITS))
    fit=fit_absolute(cal_donor,cal_host,fresh)
    s32,n32=evaluate(host,panel,target_donor,[mx.array(v) for v in fresh],[mx.array(v) for v in fit],
                     cut=13,borrowed=(old[0],old[1],old[4],old[5]))
    arms={k:dict(full_vocabulary=metrics(v,full=True),forced=metrics(v,full=False))
          for k,v in (("14b",s14),("32b",s32))}
    chosen=arms["32b"]["full_vocabulary"]
    eligible=(s32["native"][:,:,0]==1).all(1)
    n=int(eligible.sum())
    contrast=None
    if n:
        trained=(s32["transfer"][eligible,:,1]==1).all(1)
        reused=(s32["borrowed"][eligible,:,1]==1).all(1)
        draws=np.random.default_rng(501306).integers(n,size=(5000,n))
        contrast=interval(trained.astype(int)-reused,draws)
    qualified=chosen["qualifies"] and chosen["reversals"]["paired"]>=.8*n
    np.savez_compressed(out/"14b.npz",norms=n14,**s14)
    np.savez_compressed(out/"32b.npz",norms=n32,**s32)
    np.savez_compressed(out/"map-32b.npz",donor=fresh[0],host=fresh[1],orthogonal=fresh[2],
                       paired=fresh[3],absolute=fit[0],bias=fit[1])
    (out/"programs.json").write_text(json.dumps(dict(host=panel,donor=source),separators=(",",":"))+"\n")
    result=dict(decision="scales_to_32b" if qualified else "unresolved",
                models={"donor":MODELS["donor"],"14b":MODELS["host"],"32b":HOST_32},
                groups=len(panel),old_map_sha256=digest,arms=arms,shared=shared_scores(s14,s32),
                fitted_minus_borrowed=contrast,fit_advantage=bool(contrast and contrast["interval"][0]>=.2))
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
