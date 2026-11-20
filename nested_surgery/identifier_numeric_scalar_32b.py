"""Estimate one transfer gain from answer-free assignment-state differences."""
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
from identifier_computation import programs
from identifier_numeric_raw_32b import PRIOR as MATCHED_JOB,evaluate
from identifier_numeric_scale import HOST_32,PRIOR as MAP_JOB,archived_maps
from identifier_transfer import gather
from large_opener import MODELS,checkpoint_files,model_files

RAW_JOB="qwen-one-pass-raw-32b"
RAW_REVISION="92686641e3d180b0d1ffcd86359688b39eedf4ec"


def calibrated_gain(states14,states32):
    distances=[np.linalg.norm(x[1::2]-x[::2],axis=1) for x in (states14,states32)]
    assert all(np.isfinite(x).all() and (x>0).all() for x in distances)
    value=float(np.median(distances[1]/distances[0]))
    return value,np.column_stack(distances)


def summarize(current,raw,matched):
    base=[(d["native"][:,:,0]==1).all(1) for d in (current,raw,matched)]
    shared=np.logical_and.reduce(base)
    n=int(shared.sum())
    arms={"gain":current["borrowed_raw"],"random":current["norm_matched"],
          "raw":raw["borrowed_raw"],"fitted_magnitude":matched["borrowed"]}
    hits={m:(z[shared,:,1]==1).all(1) for m,z in arms.items()}
    total={k:int(v.sum()) for k,v in hits.items()}
    differences={}
    if n:
        draws=np.random.default_rng(278042).integers(n,size=(5000,n))
        differences={other:interval(hits["gain"].astype(int)-hits[other],draws)
                     for other in ("raw","random","fitted_magnitude")}
    qualified=(n>=100 and total["gain"]>=.75*n and differences["raw"]["interval"][0]>=.20
               and differences["fitted_magnitude"]["interval"][0]>=-.10
               and differences["random"]["interval"][0]>0)
    return dict(decision="answer_free_gain" if qualified else "unresolved",eligible=n,
                native_by_run=[int(a.sum()) for a in base],reversals=total,gain_minus=differences)


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    raw=root/RAW_JOB
    receipt=json.loads((raw/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]==RAW_REVISION
    matched=root/MATCHED_JOB
    payload=(matched/"programs.json").read_bytes()
    panel=json.loads(payload)
    paths={"donor":model_files("donor"),"14b":model_files("host"),"32b":checkpoint_files(*HOST_32)}
    tok=Tokenizer.from_file(str(paths["32b"]/"tokenizer.json"))
    cal=programs(tok,128,123000,computed=False)
    assert cal==programs(Tokenizer.from_file(str(paths["14b"]/"tokenizer.json")),128,123000,computed=False)
    donor,_=load_model(paths["donor"],strict=True)
    state=gather(donor,panel["donor"],SITES["donor"])
    del donor
    gc.collect()
    host,_=load_model(paths["14b"],strict=True)
    h14=gather(host,cal,10)
    del host
    gc.collect()
    host,_=load_model(paths["32b"],strict=True)
    assert len(host.model.layers)==64
    h32=gather(host,cal,13)
    gain,distances=calibrated_gain(h14,h32)
    print(f"Answer-free gain: {gain:.6f}",flush=True)
    mapping,digest=archived_maps(root/MAP_JOB)
    scores,norms=evaluate(host,panel["host"],state,mapping,gain=gain)
    with np.load(raw/"raw-32b.npz") as a,np.load(matched/"32b.npz") as b:
        summary=summarize(scores,a,b)
    np.savez_compressed(out/"scalar-32b.npz",norms=norms,distances=distances,**scores)
    (out/"calibration.json").write_text(json.dumps(cal,separators=(",",":"))+"\n")
    report=dict(models={"donor":MODELS["donor"],"14b":MODELS["host"],"32b":HOST_32},groups=len(panel["host"]),
                gain=gain,map_digest=digest,prompts_digest=sha256(payload).hexdigest(),**summary)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
