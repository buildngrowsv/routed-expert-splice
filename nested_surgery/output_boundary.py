"""Cross exact and refitted donor output at one position in the trained 14B host."""
from hashlib import file_digest
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_calibration import crossed
from bridge_input import interval, signal
from large_graft import HOST_BLOCK, encode, evaluate, finish, inputs, model_files
from large_opener import MODELS
from opener_programs import programs


def run():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    graft = runs/"large-qwen-recalibrated-graft"
    assert json.loads((graft/"result.json").read_text())["decision"]=="stop"
    archived = runs/"large-qwen-block27-graft/interfaces.safetensors"
    refit_path = runs/"large-qwen-paired-input-calibration/calibrated-maps.safetensors"
    trained = graft/"block28-refit.safetensors"
    hashes = {}
    for k, path in (("output_archive",archived),("refit_input",refit_path),("receiver",trained)):
        with path.open("rb") as stream:
            hashes[k] = file_digest(stream,"sha256").hexdigest()
    previous = json.loads((graft/"inputs.json").read_text())["hashes"]
    assert hashes["output_archive"]==previous["archived"] and hashes["refit_input"]==previous["refit"]
    refit = mx.load(str(refit_path))
    wi,bi = refit["ordinary_w"],refit["ordinary_b"]
    original = mx.load(str(archived))
    wo,bo = original["coder_out"],original["coder_bias"]
    del original,refit
    rows = programs(256,start=70000)
    fresh = {row["prompt"] for row in rows}
    for old in ("large-qwen-block27-graft","large-qwen-native-vs-bridged-input",
                "large-qwen-paired-input-calibration","large-qwen-recalibrated-graft"):
        history = json.loads((runs/old/"programs.json").read_text())
        assert fresh.isdisjoint(row["prompt"] for row in history)
    paths = {k:model_files(k) for k in ("host","donor")}
    tokenizer = {k:Tokenizer.from_file(str(v/"tokenizer.json")) for k,v in paths.items()}
    host,_ = load_model(paths["host"],strict=True)
    donor,_ = load_model(paths["donor"],strict=True)
    host.model.layers[HOST_BLOCK+1].load_weights(str(trained))
    native = host.model.layers[HOST_BLOCK].mlp
    expert = donor.model.layers[-1].mlp
    panels = {k:[] for k in ("donor_native","host_native","host_exact_donor","host_refit_donor")}
    for offset in range(0,len(rows),12):
        sample = rows[offset:offset+12]
        ids,pos,choices,_ = encode(sample,tokenizer["host"])
        donor_ids,donor_pos,donor_choices,_ = encode(sample,tokenizer["donor"])
        np.testing.assert_array_equal(ids,donor_ids)
        np.testing.assert_array_equal(np.array(pos),np.array(donor_pos))
        np.testing.assert_array_equal(np.array(choices),np.array(donor_choices))
        ids = mx.array(ids)
        n_groups = len(ids)//3
        last = (mx.arange(len(ids)),pos)
        shape = (n_groups,3,4)
        r,u = inputs(host,ids,HOST_BLOCK)
        donor_r,donor_u = inputs(donor,ids,len(donor.model.layers)-1)
        authoritative = expert(donor_u[last])
        refit_donor = expert((u[last].astype(mx.float32)@wi+bi).astype(mx.bfloat16))
        panels["donor_native"].append(crossed(donor,donor_r[last],authoritative,choices))
        native_update = native(u)
        host_branches = {
            "host_native":native_update[last],
            "host_exact_donor":(authoritative.astype(mx.float32)@wo+bo).astype(mx.bfloat16),
            "host_refit_donor":(refit_donor.astype(mx.float32)@wo+bo).astype(mx.bfloat16),
        }
        at = (mx.arange(ids.shape[1])[None,:]==pos[:,None])[...,None]
        groups = np.arange(len(ids)).reshape(n_groups,3)
        for key,contribution in host_branches.items():
            scores = []
            for source in range(3):
                ix = mx.array(np.repeat(groups[:,source],3))
                changed = mx.where(at,contribution[ix][:,None,:],native_update)
                scores.append(np.asarray(finish(host,r+changed,pos,choices)).reshape(shape))
            cube = np.stack(scores,axis=2)
            panels[key].append(cube)
            if offset==0 and key=="host_native":
                clean = evaluate(host,encode(sample,tokenizer["host"]),native)
                diagonal = cube[:,np.arange(3),np.arange(3),:].reshape(len(ids),4)
                np.testing.assert_allclose(clean,diagonal,atol=.25,rtol=.02)
                np.testing.assert_array_equal(clean.argmax(1),diagonal.argmax(1))
    cubes = {k:np.concatenate(v) for k,v in panels.items()}
    assert all(np.isfinite(v).all() for v in cubes.values())
    effects = {k:signal(v) for k,v in cubes.items()}
    means = {k:v.mean(1) for k,v in effects.items()}
    n = len(rows)//3
    idx = np.random.default_rng(745029).integers(n,size=(5000,n))
    source = means["donor_native"]
    half = {k:interval(means[k]-.5*source,idx) for k in ("host_exact_donor","host_refit_donor")}
    outcome = "unresolved"
    if interval(source,idx)["interval"][0]>0:
        exact,refit = [half[k]["interval"] for k in ("host_exact_donor","host_refit_donor")]
        if exact[1]<0:
            outcome = "output_below_margin"
        elif exact[0]>0 and refit[1]<0:
            outcome = "input_gap_in_host"
        elif refit[0]>0:
            outcome = "refitted_local_signal"
    solved = {k:int((v[:,np.arange(3),np.arange(3),:].argmax(-1)==np.arange(3)).all(1).sum()) for k,v in cubes.items()}
    np.savez_compressed(out/"crossed-scores.npz",**cubes)
    (out/"programs.json").write_text(json.dumps(rows,indent=2)+"\n")
    report = dict(decision=outcome,models=MODELS,hashes=hashes,groups=n,
                  effects={k:interval(v,idx) for k,v in means.items()},
                  pair_means={k:v.mean(0).tolist() for k,v in effects.items()},
                  host_minus_half_native_donor=half,complete=solved)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    run()
