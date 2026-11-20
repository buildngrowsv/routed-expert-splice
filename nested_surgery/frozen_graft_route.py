"""Test whether sparse score fusion improves the intact specialist on fresh repairs."""
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

from large_graft import HOST_BLOCK, encode, finish, inputs
from large_opener import MODELS, model_files
from learned_output_port import GROUPS_PER_BATCH, collect, panel, project
from learned_output_port_decoys import PRIOR, REVISION, rewrite
from opener_programs import programs

HERE = Path(__file__).resolve().parent
DECISION_WEIGHT = .25
N = 768


def margin(values):
    peaks = np.sort(values, axis=-1)
    return peaks[..., -1] - peaks[..., -2]


def gate(donor, threshold):
    return margin(donor) <= threshold


def select(donor, graft, mask):
    weighted = donor + DECISION_WEIGHT * (graft - donor)
    return np.where(mask[..., None], weighted, donor)


def outcomes(donor, graft, threshold):
    donor, graft = donor.astype(float), graft.astype(float)
    calls = gate(donor, threshold)
    combined = select(donor, graft, calls)
    gold = np.arange(3)
    correct = {
        "intact": (donor.argmax(-1) == gold).all(axis=-1),
        "graft": (graft.argmax(-1) == gold).all(axis=-1),
        "always_blend": (select(donor, graft, np.ones_like(calls)).argmax(-1) == gold).all(axis=-1),
        "hard_route": (np.where(calls[..., None], graft, donor).argmax(-1) == gold).all(axis=-1),
        "confidence_blend": (combined.argmax(-1) == gold).all(axis=-1),
    }
    rng = np.random.default_rng(231780)
    rate = np.zeros(len(donor))
    n_calls = int(calls.sum())
    for _ in range(32):
        ordered = rng.permutation(calls.size)
        random_calls = np.zeros(calls.size, dtype=bool)
        random_calls[ordered[:n_calls]] = True
        observed = select(donor, graft, random_calls.reshape(calls.shape)).argmax(-1)
        rate += (observed == gold).all(axis=-1) / 32
    draws = np.random.default_rng(87515).integers(len(donor), size=(5000, len(donor)))
    routed = correct["confidence_blend"]
    comparisons = {"intact": correct["intact"], "random_blend": rate}
    paired = {}
    for label, v in comparisons.items():
        diff = routed.astype(float) - v
        paired[label] = dict(mean=float(diff.mean()),
                             interval=np.percentile(diff[draws].mean(axis=1), [2.5, 97.5]).tolist())
    result = dict(complete={k: int(v.sum()) for k, v in correct.items()},
                  mean_random_complete=float(rate.sum()), paired=paired,
                  recovered=int((routed & ~correct["intact"]).sum()),
                  lost=int((~routed & correct["intact"]).sum()),
                  calls=n_calls, prompts=int(calls.size), call_fraction=float(calls.mean()))
    return result, calls


def check_subset(host, rows, tokens, projected, calls, expected, weights):
    """Execute only the selected host prompts in the first eight input batches."""
    calls = calls.reshape(-1)
    expected = expected.reshape(-1, 4)
    errors, count = 0, 0
    size = 3 * GROUPS_PER_BATCH
    for batch in range(min(8, len(projected))):
        start = batch * size
        subset = np.flatnonzero(calls[start:start+size])
        if not len(subset):
            continue
        data = rows[start:start+size]
        ids, pos, options, _ = encode(data, tokens["host"])
        indices = mx.array(subset, dtype=mx.int32)
        pos = pos[indices]
        r, u = inputs(host, mx.array(ids[subset]), HOST_BLOCK)
        state = r + host.model.layers[HOST_BLOCK].mlp(u)
        delta = projected[batch][indices] @ weights["port"] + weights["bias"]
        changed = state.at[mx.arange(len(subset)), pos].add(delta.astype(state.dtype))
        sparse = np.asarray(finish(host, changed, pos, options))
        errors += int((sparse.argmax(1) != expected[start+subset].argmax(1)).sum())
        count += len(subset)
    return dict(calls=count, candidate_differences=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    small = parser.parse_args().smoke
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    calibration = HERE / "learned_output_port_decoys_study/results"
    result = json.loads((calibration / "result.json").read_text())
    assert result["decision"] == "frozen_donor_adds_decoy_repairs"
    calibration_raw = calibration / "decoys.npz"
    with np.load(calibration_raw) as z:
        threshold = float(np.quantile(margin(z["intact_donor"]), .25))
    assert threshold == .5625
    prior = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs" / PRIOR
    receipt = json.loads((prior / "receipt.json").read_text())
    assert (receipt["revision"], receipt["status"], receipt["exit"]) == (REVISION, "done", 0)
    path = prior / "ports.safetensors"
    assert sha256(path.read_bytes()).hexdigest() == result["port_sha256"]
    saved = mx.load(str(path))
    basis = {k: np.asarray(saved[f"basis_donor_feature_{k}"]) for k in ("center", "directions", "scale")}
    weights = {k: saved[f"learner_donor_{k}"] for k in ("port", "bias")}
    models = {k: model_files(k) for k in MODELS}
    tokens = {k: Tokenizer.from_file(str(p / "tokenizer.json")) for k, p in models.items()}
    host, _ = load_model(models["host"], strict=True)
    coder, _ = load_model(models["donor"], strict=True)
    programs_base = programs(4 if small else N, start=85000)
    matched = {name: rewrite(programs_base, mode) for name, mode in (("neutral", False), ("decoys", True))}
    out = None if small else Path(os.environ["STUDIO_JOB_DIR"])
    results = {}
    for name, rows in matched.items():
        print(f"Scoring the {name} layout", flush=True)
        batches = collect(host, coder, rows, tokens)
        donor = np.concatenate([b.donor_logits for b in batches]).reshape(-1, 3, 4)
        source = project(batches, "donor_feature", basis)
        graft = panel(host, batches, source, weights)
        summary, calls = outcomes(donor, graft, threshold)
        summary["actual_subset_audit"] = check_subset(host, rows, tokens, source, calls, graft, weights)
        results[name] = summary
        if out is not None:
            np.savez_compressed(out / f"{name}.npz", donor=donor, graft=graft, calls=calls)
    neutral, decoys = results["neutral"], results["decoys"]
    effects = decoys["paired"]
    good = (effects["intact"]["mean"] >= .03 and effects["intact"]["interval"][0] > 0
            and effects["random_blend"]["mean"] >= .02 and effects["random_blend"]["interval"][0] > 0
            and neutral["paired"]["intact"]["interval"][0] >= -.02
            and max(d["call_fraction"] for d in results.values()) <= .33
            and all(d["actual_subset_audit"]["candidate_differences"] == 0 for d in results.values()))
    report = dict(decision="sparse_fusion_adds_repairs" if good else "unresolved", layouts=results, models=MODELS,
                  groups=len(programs_base)//3, weight=DECISION_WEIGHT, threshold=threshold,
                  calibration_sha256=sha256(calibration_raw.read_bytes()).hexdigest(),
                  port_sha256=result["port_sha256"], runtime_s=time.perf_counter()-started,
                  peak_mlx_bytes=mx.get_peak_memory())
    if out is None:
        assert all(x["actual_subset_audit"]["candidate_differences"] == 0 for x in results.values())
        report["decision"] = "smoke_complete"
        print(json.dumps(report))
        if "STUDIO_JOB_DIR" in os.environ:
            (Path(os.environ["STUDIO_JOB_DIR"]) / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        (out / "programs.json").write_text(json.dumps(matched, separators=(",", ":")) + "\n")
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
