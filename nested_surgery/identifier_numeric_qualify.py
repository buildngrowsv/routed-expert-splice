"""Test native arithmetic competence before paying for two-site surgery."""

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES
from identifier_computation import programs
from identifier_numeric_composition import MAP_JOB, evaluate, host_panel, load_prior, summarize
from identifier_transfer import gather, transform
from large_opener import MODELS, model_files


def native_groups(scores):
    labels = np.arange(4)[None,:,None]
    selected = np.take_along_axis(scores,labels,axis=2)[...,0]
    others = np.where(np.arange(4)[None,None,:]==labels,-np.inf,scores).max(2)
    return (selected>others).all(1)


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {k:model_files(k) for k in SITES}
    tokenizer = Tokenizer.from_file(str(paths["host"]/"tokenizer.json"))
    qualification = host_panel(tokenizer,128,92000,simple=True)
    host,_ = load_model(paths["host"],strict=True)
    baseline = evaluate(host,qualification)["native"]
    assert baseline.shape==(128,4,4) and np.isfinite(baseline).all()
    eligible = int(native_groups(baseline).sum())
    result = dict(decision="benchmark_below_floor",models=MODELS,screen_groups=128,screen_eligible=eligible)
    np.savez_compressed(out/"qualification.npz",native=baseline)
    (out/"qualification.json").write_text(json.dumps(qualification,separators=(",",":"))+"\n")
    print(f"Native qualification {eligible}/128 complete groups",flush=True)
    if eligible>=20:
        del host
        gc.collect()
        panel = host_panel(tokenizer,512,93000,simple=True)
        prior = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/MAP_JOB
        axes,digest = load_prior(prior)
        donor_tok = Tokenizer.from_file(str(paths["donor"]/"tokenizer.json"))
        donor,_ = load_model(paths["donor"],strict=True)
        sources,changes,norms = {},{},{}
        for side,start in (("left",94000),("right",95000)):
            src = programs(donor_tok,len(panel),start,computed=False,values_by_pair=[r[side] for r in panel])
            sources[side] = src
            state = gather(donor,src,SITES["donor"]).astype(np.float32)
            edits = transform(mx.array(state),*axes)
            changes[side] = [np.array(v).reshape(len(panel),2,-1) for v in edits]
            norms[side] = np.stack([np.linalg.norm(v,axis=2) for v in changes[side]],axis=-1)
        del donor
        gc.collect()
        host,_ = load_model(paths["host"],strict=True)
        scores = evaluate(host,panel,changes)
        assert all(v.shape==(512,4,4) and np.isfinite(v).all() for v in scores.values())
        result.update(**summarize(scores,panel),groups=len(panel),map_digest=digest)
        np.savez_compressed(out/"composition.npz",left_norms=norms["left"],right_norms=norms["right"],**scores)
        (out/"programs.json").write_text(json.dumps(dict(host=panel,donor=sources),separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
