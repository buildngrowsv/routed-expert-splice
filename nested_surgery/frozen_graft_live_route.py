"""Execute only requested 14B inputs and measure the synchronized inference cost."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import read_logits
from frozen_graft_route import DECISION_WEIGHT, gate, select
from large_graft import HOST_BLOCK, encode, finish, inputs
from large_opener import MODELS, model_files

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "frozen_graft_route_confirmation_study/results"
SIZE = 12
REPEATS = 3
TIMED_BATCHES = 32
ARMS = ("standalone", "always_blend", "conditional")


def run_batch(host, coder, ids, positions, choices, basis, interface, threshold, arm):
    last = len(coder.model.layers) - 1
    dr, du = inputs(coder, mx.array(ids), last)
    expert = coder.model.layers[last].mlp(du[mx.arange(len(ids)), positions])
    donor = read_logits(coder, coder.model.norm(dr[mx.arange(len(ids)), positions] + expert), choices).astype(float)
    selected = gate(donor, threshold) if arm == "conditional" else np.full(len(ids), arm == "always_blend")
    subset = np.flatnonzero(selected)
    if not len(subset):
        return donor.argmax(1), selected
    on_device = mx.array(subset, dtype=mx.int32)
    pos = positions[on_device]
    r, u = inputs(host, mx.array(ids[subset]), HOST_BLOCK)
    h = r + host.model.layers[HOST_BLOCK].mlp(u)
    features = np.asarray(expert[on_device].astype(mx.float32))
    mu, rotation, scale = basis
    source = mx.array((features - mu) @ rotation / scale)
    addition = (source @ interface["port"] + interface["bias"]).astype(h.dtype)
    h = h.at[mx.arange(len(subset)), pos].add(addition)
    graft = np.asarray(finish(host, h, pos, choices)).astype(float)
    output = donor.copy()
    output[subset] += DECISION_WEIGHT * (graft - donor[subset])
    return output.argmax(1), selected


def timing(elapsed, seed):
    medians = np.median(elapsed, axis=1)
    gains = medians[:, 1] / medians[:, 2]
    samples = np.random.default_rng(seed).integers(len(gains), size=(5000, len(gains)))
    return dict(median_ms=(np.median(elapsed,axis=(0,1))/1e6).tolist(),
                speedup_over_always=float(np.median(gains)),
                speedup_interval=np.percentile(np.median(gains[samples],axis=1),[2.5,97.5]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    small = parser.parse_args().smoke
    started = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    archive = json.loads((ARCHIVE / "result.json").read_text())
    assert archive["decision"] == "sparse_fusion_adds_repairs"
    threshold = archive["threshold"]
    assert DECISION_WEIGHT == archive["weight"]
    path = HERE / "learned_output_port_study/results/ports.safetensors"
    assert sha256(path.read_bytes()).hexdigest() == archive["port_sha256"]
    saved = mx.load(str(path))
    basis = tuple(np.asarray(saved[f"basis_donor_feature_{key}"]) for key in ("center","directions","scale"))
    interface = {key:saved[f"learner_donor_{key}"] for key in ("port","bias")}
    programs = json.loads((ARCHIVE / "programs.json").read_text())
    paths = {k:model_files(k) for k in MODELS}
    tokenizer = {k:Tokenizer.from_file(str(v/"tokenizer.json")) for k,v in paths.items()}
    host, _ = load_model(paths["host"], strict=True)
    coder, _ = load_model(paths["donor"], strict=True)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    result = {}
    evidence = {}
    for case, rows in programs.items():
        with np.load(ARCHIVE / f"{case}.npz") as source:
            d, g = (source[k].astype(float) for k in ("donor","graft"))
            expected_calls = source["calls"].ravel()
        expected = select(d,g,expected_calls.reshape(d.shape[:2])).argmax(-1).ravel()
        if small:
            rows = rows[:SIZE]
        indices = np.sort(np.random.default_rng(88143).choice(len(rows)//SIZE, min(TIMED_BATCHES,len(rows)//SIZE), replace=False))
        packed = [encode(rows[i:i+SIZE],tokenizer["host"])[:3] for i in range(0,len(rows),SIZE)]
        for i in indices:
            ids, pos, choices = encode(rows[i*SIZE:(i+1)*SIZE], tokenizer["donor"])[:3]
            np.testing.assert_array_equal(ids, packed[i][0])
            np.testing.assert_array_equal(np.asarray(pos), np.asarray(packed[i][1]))
            np.testing.assert_array_equal(np.asarray(choices), np.asarray(packed[i][2]))
            mx.eval(*packed[i][1:])
        actual, masks = [], []
        durations, timed_choices = [], []
        for i, args in enumerate(packed):
            fn = lambda mode: run_batch(host, coder, *args, basis, interface, threshold, mode)
            chosen, used = fn("conditional")
            actual.extend(chosen)
            masks.extend(used)
            if i in indices:
                warm = [fn(arm)[0] for arm in ARMS]
                times = np.zeros((REPEATS,len(ARMS)), dtype=np.int64)
                for rep in range(REPEATS):
                    for k in np.roll(np.arange(len(ARMS)), i+rep):
                        t0 = time.perf_counter_ns()
                        got, _ = fn(ARMS[k])
                        times[rep,k] = time.perf_counter_ns()-t0
                        np.testing.assert_array_equal(got, warm[k])
                durations.append(times)
                timed_choices.append(warm)
            if (i+1) % 96 == 0:
                print(f"{case}: executed actual sparse route on {(i+1)*SIZE}/{len(rows)} prompts",flush=True)
        actual = np.asarray(actual)
        masks = np.asarray(masks)
        misses = (actual != expected[:len(actual)]).reshape(-1,3).any(1)
        n = len(actual)//3
        eligible = (actual.reshape(-1,3) == np.arange(3)).all(1)
        native = (d.argmax(-1)[:n] == np.arange(3)).all(1)
        delta = eligible.astype(int) - native
        draws = np.random.default_rng(18851).integers(n,size=(5000,n))
        elapsed = np.stack(durations)
        result[case] = dict(n=n, complete=int(eligible.sum()), native=int(native.sum()), mismatched_groups=int(misses.sum()),
                            call_changes=int((masks != expected_calls[:len(masks)]).sum()),
                            actual_calls=int(masks.sum()), call_fraction=float(masks.mean()),
                            lift=float(delta.mean()), interval=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
                            **timing(elapsed,15239))
        evidence.update({f"{case}_actual":actual, f"{case}_called":masks, f"{case}_timed_indices":indices,
                         f"{case}_durations_ns":elapsed, f"{case}_timed_winners":np.stack(timed_choices)})
    decoy, neutral = result["decoys"], result["neutral"]
    passes = (decoy["speedup_over_always"]>=1.5 and decoy["speedup_interval"][0]>1
              and decoy["lift"]>=.03 and decoy["interval"][0]>0 and neutral["interval"][0]>=-.02
              and all(r["mismatched_groups"] <= .005*r["n"] and r["call_fraction"]<=.33 for r in result.values()))
    if small:
        assert all(r["mismatched_groups"]==0 and r["call_changes"]==0 for r in result.values())
        decision = "smoke_complete"
    else:
        decision = "live_route_confirms_speed_and_quality" if passes else "unresolved"
        np.savez_compressed(out/"live-route.npz",**evidence)
    report = dict(decision=decision, layouts=result, arms=ARMS, repeats=REPEATS,
                  seconds=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
