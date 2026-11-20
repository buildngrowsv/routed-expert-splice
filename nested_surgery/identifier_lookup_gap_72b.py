"""Test an exact token-gap control for late-key donor steering."""
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST
from identifier_numeric_route_72b import evaluate, good_groups
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

FRESH = "qwen-late-key-donor-fresh-72b"
REVISION = "ad2451d54e4b6c70aa73749686402a0f317a4897"


def exact_gap(tokenizer, early, late):
    result = []
    for before, after in zip(early, late, strict=True):
        gap = len(before["ids"][0]) - before["site"]
        prompts = after["prompts"]
        for length in range(1, 65):
            comment = "    #" + " note" * length + "\n"
            padded = []
            for text in prompts:
                lines = text.splitlines(keepends=True)
                assert lines[2].startswith("    mode_") and lines[3].startswith("    return table[")
                padded.append("".join(lines[:3] + [comment] + lines[3:]))
            tokens = [tokenizer.encode(p, add_special_tokens=False).ids for p in padded]
            if all(len(t)-after["site"] == gap for t in tokens):
                break
        else:
            raise ValueError("No comment matched the token gap")
        site = after["site"]
        assert all(t[:site + 1] == u[:site + 1] for t, u in zip(tokens, after["ids"], strict=True))
        assert [i for i, pair in enumerate(zip(*tokens, strict=True)) if pair[0] != pair[1]] == [site]
        for text, answer, seq, answer_id in zip(padded, after["answers"], tokens, after["choices"], strict=True):
            exec(compile(text + answer + "\n", "<matched token gap>", "exec"), {})
            assert tokenizer.encode(text+answer, add_special_tokens=False).ids == seq + [answer_id]
        result.append(dict(after, prompts=padded, ids=tokens, matched_gap=gap))
    return result


def summarize(scores):
    native = {m: good_groups(z["native"]) for m, z in scores.items()}
    shared = np.logical_and.reduce(list(native.values()))
    n = int(shared.sum())
    switches, preserves = {}, {}
    for layout, data in scores.items():
        for arm in ("donor", "whole", "orthogonal"):
            label = f"{layout}_{arm}"
            rank = data[arm].reshape(-1,3,2,4)[shared]
            switches[label] = (rank[:, :2, :, 1] == 1).all(axis=(1,2))
            preserves[label] = int((rank[:, 2, :, 0] == 1).all(axis=1).sum())
    effect = {}
    if n:
        draws = np.random.default_rng(764554).integers(n,size=(5000,n))
        for label, a, b in (("gap_minus_early", "gap_donor", "early_donor"),
                            ("gap_minus_late", "gap_donor", "late_donor"),
                            ("gap_minus_orthogonal", "gap_donor", "gap_orthogonal")):
            effect[label] = interval(switches[a].astype(int)-switches[b],draws)
    rates = {m:int(x.sum()) for m,x in switches.items()}
    decision = "unresolved"
    if n >= 100 and preserves["gap_donor"] >= .95*n:
        gain, near, control = (effect[k] for k in ("gap_minus_early","gap_minus_late","gap_minus_orthogonal"))
        if (min(rates[m] for m in ("gap_donor", "gap_whole")) >= .8*n and gain["mean"] >= .15
                and gain["interval"][0] > 0 and near["interval"][0] >= -.10
                and control["mean"] >= .20 and control["interval"][0] > 0):
            decision = "late_key_gain_survives_matched_gap"
        elif near["mean"] <= -.15 and near["interval"][1] < 0:
            decision = "neutral_gap_reduces_late_key_control"
    return dict(decision=decision, shared=n, native={k:int(v.sum()) for k,v in native.items()},
                switches=rates, preserved=preserves, paired=effect)


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    prior = verified_result(root/FRESH, REVISION, "late_key_donor_transfer")
    fit = verified_result(root/ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    payload = (root/FRESH/"programs.json").read_bytes()
    assert sha256(payload).hexdigest() == prior["programs_digest"]
    prompts = json.loads(payload)
    host_path = checkpoint_files(*HOST)
    tokenizer = Tokenizer.from_file(str(host_path/"tokenizer.json"))
    rows = exact_gap(tokenizer, prompts["early"], prompts["late"])
    mapping, map_hash = archived_maps(root/MAP_JOB)
    assert map_hash == prior["map_digest"] == fit["map_digest"]
    donor_axis, _, _, _, affine, intercept = mapping
    donor, _ = load_model(model_files("donor"), strict=True)
    state = mx.array(gather(donor,prompts["donor"],SITES["donor"]).astype(np.float32))
    vectors = (state @ donor_axis @ affine + intercept).reshape(len(rows)//3,2,-1)
    targets = mx.repeat(vectors,3,axis=0).reshape(len(rows)*2,-1)
    mx.eval(targets)
    del donor
    gc.collect()
    matrix = root/ALIGNMENT_JOB/"alignment.npz"
    mat_hash = sha256(matrix.read_bytes()).hexdigest()
    assert mat_hash == prior["alignment_digest"]
    with np.load(matrix) as f:
        geometry = tuple(mx.array(f[k].astype(np.float32))
                         for k in ("write","read","random","center_input","center_target"))
    model, _ = load_model(host_path,strict=True)
    gap, norms = evaluate(model,rows,targets,geometry)
    np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
    arms = {"gap":gap}
    hashes = {}
    for layout in ("early","late"):
        source = root/FRESH/f"{layout}.npz"
        hashes[layout] = sha256(source.read_bytes()).hexdigest()
        with np.load(source) as f:
            arms[layout] = {k:f[k] for k in ("native","donor","whole","orthogonal")}
    result = summarize(arms)
    np.savez_compressed(out/"gap.npz",norms=norms,**gap)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    result.update(map_digest=map_hash,alignment_digest=mat_hash,original_programs_digest=prior["programs_digest"],
                  original_ranks_digests=hashes,cut=CUT,models={"donor":MODELS["donor"],"recipient":HOST},
                  runtime_s=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
