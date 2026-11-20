"""Bridge a narrower interface into 72B using native state contrasts only."""
import argparse
import gc
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
from identifier_baseline import SITES, encoded_batches, prefix
from identifier_binding import patch
from identifier_computation import programs
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_scale import HOST_32, PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

HOST = ("Qwen/Qwen2.5-72B", "efba10c8e54e91e0d9570ab5f7b51a958474d4cb")
REFERENCE = "qwen-one-pass-affine-32b"
SCALAR = "qwen-one-pass-scalar-32b"
MODES = ("native", "stitch", "random", "whole")


def alignment(narrow, wide, axes):
    projected = narrow @ axes
    x = projected[1::2] - projected[::2]
    y = wide[1::2] - wide[::2]
    gram = x.T @ x
    penalty = .02 * np.trace(gram) / len(gram)
    write = np.linalg.solve(gram + penalty * np.eye(len(gram)), x.T @ y).T
    read = np.linalg.pinv(write).T
    assert np.linalg.matrix_rank(write) == len(gram)
    np.testing.assert_allclose(read.T @ write, np.eye(len(gram)), atol=1e-8)
    q, _ = np.linalg.qr(write)
    other = np.random.default_rng(736925).normal(size=write.shape)
    orthogonal, _ = np.linalg.qr(other - q @ (q.T @ other))
    center_input, center_target = wide.mean(0), projected.mean(0)
    return dict(x=x, y=y, write=write, read=read, random=orthogonal,
                center_input=center_input, center_target=center_target, penalty=penalty)


def evaluate(model, rows, donor_targets, geometry, cut):
    keys = ("write", "read", "random", "center_input", "center_target")
    write, read, random, cx, cy = (mx.array(geometry[key].astype(np.float32)) for key in keys)
    report = {m: [] for m in MODES}
    norms = []
    seen = 0
    for ids, sites, end, choices, _ in encoded_batches(rows):
        h = prefix(model, ids, cut)
        state = h[mx.arange(len(sites)), sites].astype(mx.float32)
        targets = donor_targets[seen:seen + len(sites)]
        partner = mx.array(np.arange(len(sites)) ^ 1)
        desired = targets[partner] - ((state - cx) @ read + cy)
        edit = desired @ write.T
        sham = desired @ random.T
        norm = mx.linalg.norm(edit, axis=1, keepdims=True)
        sham = sham * (norm / mx.maximum(mx.linalg.norm(sham, axis=1, keepdims=True), 1e-12))
        norms.append(np.column_stack([np.array(norm).ravel(), np.array(mx.linalg.norm(sham, axis=1))]))
        report["native"].extend(whole_vocabulary(model, h, end, cut, choices))
        for mode, updated in (("stitch", state + edit), ("random", state + sham), ("whole", state[partner])):
            report[mode].extend(whole_vocabulary(model, patch(h, sites, updated), end, cut, choices))
        seen += len(sites)
        if seen % 128 == 0:
            print(f"Scored {seen // 2}/{len(rows)} pairs", flush=True)
    scores = {m: np.array(v) for m, v in report.items()}
    doses = np.concatenate(norms)
    assert all(v.shape == (len(rows), 2, 4) and np.isfinite(v).all() for v in scores.values())
    np.testing.assert_allclose(doses[:, 0], doses[:, 1], rtol=1e-5, atol=1e-6)
    return scores, doses


def summarize(scores, prior):
    eligible = (scores["native"][:, :, 0] == 1).all(1)
    shared = eligible & (prior["native"][:, :, 0] == 1).all(1)
    hits = {m: (scores[m][:, :, 1] == 1).all(1) for m in MODES[1:]}
    hits["scalar_32b"] = (prior["borrowed_raw"][:, :, 1] == 1).all(1)
    n = int(eligible.sum())
    correct = {m: int(h[eligible].sum()) for m, h in hits.items() if m != "scalar_32b"}
    contrast = None
    if n:
        d = (hits["stitch"].astype(int) - hits["random"])[eligible]
        draws = np.random.default_rng(491205).integers(n, size=(5000, n))
        contrast = interval(d, draws)
    passed = (n >= 100 and correct["stitch"] >= .75*n and correct["whole"] >= .8*n
              and contrast["mean"] >= .2 and contrast["interval"][0] > 0)
    return dict(decision="cross_width_transfer" if passed else "unresolved",
                eligible=n, reversals=correct, stitch_minus_random=contrast,
                shared_eligible=int(shared.sum()), shared_reversals={m: int(h[shared].sum()) for m, h in hits.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient", choices=("32b", "72b"), default="72b")
    args = parser.parse_args()
    name = args.recipient
    checkpoint, width, layers, cut = (HOST_32, 5120, 64, 13) if name == "32b" else (HOST, 8192, 80, 17)
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    paths = {"receiver": checkpoint_files(*checkpoint), "14b": model_files("host"), "donor": model_files("donor")}
    tokenizers = {m: Tokenizer.from_file(str(p / "tokenizer.json")) for m, p in paths.items()}
    calibration = programs(tokenizers["14b"], 256, 125000, computed=False)
    assert calibration == programs(tokenizers["receiver"], 256, 125000, computed=False)
    old = root / REFERENCE
    receipt = json.loads((old / "receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["revision"] == "ea4014f0edf250fa8abf5d0bb52657371998aea3"
    payload = (old / "programs.json").read_bytes()
    panel = json.loads(payload)
    assert panel["host"] == programs(tokenizers["receiver"], 512, 117000, computed=True)
    mapping, map_digest = archived_maps(root / MAP_JOB)
    donor_axis, narrow_axis, _, _, affine, bias = mapping
    model, _ = load_model(paths["donor"], strict=True)
    states = mx.array(gather(model, panel["donor"], SITES["donor"]).astype(np.float32))
    target = states @ donor_axis @ affine + bias
    mx.eval(target)
    del model
    gc.collect()
    model, _ = load_model(paths["14b"], strict=True)
    narrow = gather(model, calibration, 10)
    del model
    gc.collect()
    model, _ = load_model(paths["receiver"], strict=True)
    assert len(model.model.layers) == layers and model.args.hidden_size == width
    wide = gather(model, calibration, cut)
    geometry = alignment(narrow, wide, np.array(narrow_axis, dtype=np.float64))
    np.savez_compressed(out / "alignment.npz", **geometry)
    (out / "calibration.json").write_text(json.dumps(calibration, separators=(",", ":")) + "\n")
    print("Alignment archived; scoring full-vocabulary arithmetic pairs", flush=True)
    scores, norms = evaluate(model, panel["host"], target, geometry, cut)
    with np.load(root / SCALAR / "scalar-32b.npz") as prior:
        result = summarize(scores, prior)
    np.savez_compressed(out / f"{name}.npz", **scores, norms=norms)
    if name == "32b" and result["decision"] == "cross_width_transfer":
        result["decision"] = "alignment_control_transfers"
    report = dict(models={**MODELS, "receiver": checkpoint, "32b_reference": HOST_32}, cut=cut, calibration_pairs=256,
                  groups=512, map_digest=map_digest, prompts_digest=sha256(payload).hexdigest(),
                  runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory(), **result)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
