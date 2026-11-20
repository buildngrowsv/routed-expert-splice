"""Test if calibration on paired prompts restores the frozen donor’s input signal."""
from hashlib import file_digest
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval, read_logits, signal
from large_graft import HOST_BLOCK, encode, inputs
from large_opener import MODELS, model_files
from opener_programs import programs


def fit_map(x, y, weight):
    """Dual ridge with explicit weights on each triplet's centered differences."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    assert x.shape[0] == y.shape[0] and len(x) % 3 == 0
    cx, cy = x.mean(0), y.mean(0)
    def center(v, global_mean):
        grouped = v.reshape(-1, 3, v.shape[-1])
        local = grouped.mean(1, keepdims=True)
        return (weight * (grouped-local) + local-global_mean).reshape(v.shape)
    x, y = center(x,cx), center(y,cy)
    gram = x @ x.T
    penalty = .02 * np.trace(gram) / x.shape[1]
    gram.flat[::len(gram)+1] += penalty
    matrix = x.T @ np.linalg.solve(gram,y)
    return matrix.astype(np.float32), (cy-cx@matrix).astype(np.float32)


def collect(rows, donor, host, tokens, preflight=False):
    donor_state, host_state, residual = [], [], []
    direct = None
    for i in range(0,len(rows),8):
        batch = rows[i:i+8]
        ids, pos, choices, _ = encode(batch,tokens["donor"])
        host_ids, host_pos, host_choices, _ = encode(batch,tokens["host"])
        np.testing.assert_array_equal(ids,host_ids)
        np.testing.assert_array_equal(np.array(pos),np.array(host_pos))
        np.testing.assert_array_equal(np.array(choices),np.array(host_choices))
        last = (mx.arange(len(batch)),pos)
        r,u = inputs(donor,mx.array(ids),len(donor.model.layers)-1)
        _,h = inputs(host,mx.array(host_ids),HOST_BLOCK)
        donor_state.append(u[last])
        host_state.append(h[last])
        residual.append(r[last])
        mx.eval(donor_state[-1],host_state[-1],residual[-1])
        if i==0 and preflight:
            direct = read_logits(donor,donor.model(mx.array(ids))[last],choices)
    return [mx.concatenate(v) for v in (host_state,donor_state,residual)],choices,direct


def crossed(donor, residual, contribution, choices):
    groups = np.arange(len(residual)).reshape(-1,3)
    cells = []
    for source in range(3):
        indices = mx.array(np.repeat(groups[:,source],3))
        state = donor.model.norm(residual+contribution[indices])
        cells.append(read_logits(donor,state,choices).reshape(-1,3,4))
    cube = np.stack(cells,axis=2)
    assert np.isfinite(cube).all()
    return cube


def main():
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    archived = runs/"large-qwen-block27-graft/interfaces.safetensors"
    with archived.open("rb") as stream:
        digest = file_digest(stream,"sha256").hexdigest()
    diagnosed = runs/"large-qwen-native-vs-bridged-input"
    assert json.loads((diagnosed/"result.json").read_text())["interface_sha256"] == digest
    previous = json.loads((diagnosed/"programs.json").read_text())
    calibration = programs(512,start=45000)
    rows = programs(256,start=50000)
    assert {r["prompt"] for r in rows}.isdisjoint(r["prompt"] for r in [*calibration,*previous])
    paths = {key:model_files(key) for key in ("host","donor")}
    tokens = {key:Tokenizer.from_file(str(p/"tokenizer.json")) for key,p in paths.items()}
    host,_ = load_model(paths["host"],strict=True)
    donor,_ = load_model(paths["donor"],strict=True)
    cal,_,_ = collect(calibration,donor,host,tokens)
    x,y = [np.array(v.astype(mx.float32)) for v in cal[:2]]
    del cal
    checkpoint = mx.load(str(archived))
    maps = {"archived":(checkpoint["coder_w"],checkpoint["coder_b"])}
    del checkpoint
    for name, weight in (("ordinary",1.),("weighted",4.)):
        maps[name] = tuple(mx.array(v) for v in fit_map(x,y,weight))
    (h,u,r), choices, direct = collect(rows,donor,host,tokens,preflight=True)
    expert = donor.model.layers[-1].mlp
    data = {"native":crossed(donor,r,expert(u),choices)}
    for mode,(matrix,bias) in maps.items():
        mapped = (h.astype(mx.float32)@matrix+bias).astype(mx.bfloat16)
        data[mode] = crossed(donor,r,expert(mapped),choices)
    off = read_logits(donor,donor.model.norm(r),choices)
    gold = np.array([row["gold"] for row in rows])
    np.testing.assert_array_equal(gold,np.tile(np.arange(3),len(rows)//3))
    diag = {k:v[:,np.arange(3),np.arange(3),:].reshape(-1,4) for k,v in data.items()}
    np.testing.assert_allclose(diag["native"][:8],direct,atol=.25,rtol=.02)
    np.testing.assert_array_equal(diag["native"][:8].argmax(1),direct.argmax(1))
    complete = {k:int((value.argmax(1)==gold).reshape(-1,3).all(1).sum()) for k,value in {**diag,"off":off}.items()}
    effects = {k:signal(value) for k,value in data.items()}
    means = {k:value.mean(1) for k,value in effects.items()}
    idx = np.random.default_rng(445638).integers(len(rows)//3,size=(5000,len(rows)//3))
    half = {k:interval(means[k]-.5*means["native"],idx) for k in ("archived","ordinary","weighted")}
    advantage = interval(means["weighted"]-means["ordinary"],idx)
    supported = interval(means["native"],idx)["interval"][0] > 0
    ordinary_qualifies = supported and half["ordinary"]["interval"][0] > 0
    weighted_qualifies = supported and half["weighted"]["interval"][0] > 0
    weighting_helps = weighted_qualifies and advantage["interval"][0] > 0
    decision = "stop"
    if weighted_qualifies and (weighting_helps or not ordinary_qualifies):
        decision = "weighted"
    elif ordinary_qualifies:
        decision = "ordinary"
    np.savez_compressed(out/"crossed-scores.npz",**data,off=off,gold=gold)
    mx.save_safetensors(str(out/"calibrated-maps.safetensors"),{f"{k}_{part}":value for k in ("ordinary","weighted")
                       for part,value in zip(("w","b"),maps[k])})
    (out/"programs.json").write_text(json.dumps(rows,indent=2)+"\n")
    (out/"calibration-ids.json").write_text(json.dumps([r["id"] for r in calibration],indent=2)+"\n")
    report = dict(decision=decision,complete=complete,models=MODELS,interface_sha256=digest,
                  effects={k:interval(v,idx) for k,v in means.items()}, weighting_helps=weighting_helps,
                  pairs={k:v.mean(0).tolist() for k,v in effects.items()},
                  map_minus_half_native=half,weighted_minus_ordinary=advantage)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
