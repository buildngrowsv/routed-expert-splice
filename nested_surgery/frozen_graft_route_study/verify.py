"""Audit the confidence threshold, exact call budgets and paired repair differences."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--confirmation", action="store_true")
confirmation = parser.parse_args().confirmation
if confirmation:
    P = here.parent / "frozen_graft_route_confirmation_study/results"
    job, revision = "large-qwen-frozen-graft-route-confirmation", "17fcd359c0397b0f3ca02d067a84cedb066dbc8a"
    groups, offset = 2304, 87000
else:
    P = here / "results"
    job, revision = "large-qwen-frozen-graft-route-v2", "45ffd5c9d0dca431774d7e02d677aa05a0233749"
    groups, offset = 768, 85000
receipt = json.loads((P / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    job, revision, "done", 0)
report = json.loads((P / "result.json").read_text())
assert report["groups"] == groups and report.get("first_group", 85000) == offset
calibration = here.parent / "learned_output_port_decoys_study/results/decoys.npz"
assert sha256(calibration.read_bytes()).hexdigest() == report["calibration_sha256"]
with np.load(calibration) as prior:
    old = np.sort(prior["intact_donor"], axis=2)
    threshold = np.quantile(old[:, :, 3] - old[:, :, 2], .25)
assert threshold == report["threshold"] == .5625 and report["weight"] == .25
ports = here.parent / "learned_output_port_study/results/ports.safetensors"
assert sha256(ports.read_bytes()).hexdigest() == report["port_sha256"]
programs = json.loads((P / "programs.json").read_text())
pattern = re.compile(r"'([()\[\]{}]{6})'\)\[0\]  # ([()\[\]{}]{6})")
alpha = str.maketrans("()[]{}", "abcdef")
for i, (neutral, decoy) in enumerate(zip(programs["neutral"], programs["decoys"], strict=True)):
    assert neutral["id"] == decoy["id"] == f"opener-{offset+i//3}-{i%3}"
    assert neutral["gold"] == decoy["gold"] == i%3
    altered, n = pattern.subn(lambda m: "'" + m[1].translate(alpha) + "')[0]  # " + m[2].translate(alpha), decoy["prompt"])
    assert n == 1 and altered == neutral["prompt"]
gold = np.arange(3)
draws = np.random.default_rng(87515).integers(groups, size=(5000, groups))
for label in ("neutral", "decoys"):
    with np.load(P / f"{label}.npz") as z:
        donor, graft, called = z["donor"].astype(float), z["graft"].astype(float), z["calls"]
    assert donor.shape == graft.shape == (groups, 3, 4) and called.shape == (groups, 3)
    assert np.isfinite(donor).all() and np.isfinite(graft).all()
    distance = np.diff(np.sort(donor, axis=-1), axis=-1)[..., -1]
    np.testing.assert_array_equal(distance <= threshold, called)
    bundle = .75*donor+.25*graft
    merged = np.where(called[..., None], bundle, donor)
    mask = lambda x: (x.argmax(axis=-1) == gold).all(axis=-1)
    members = dict(intact=mask(donor), graft=mask(graft), always_blend=mask(bundle),
                  hard_route=mask(np.where(called[..., None], graft, donor)), confidence_blend=mask(merged))
    stated = report["layouts"][label]
    assert {k: int(v.sum()) for k, v in members.items()} == stated["complete"]
    assert called.sum() == stated["calls"] and called.size == stated["prompts"]
    np.testing.assert_allclose(called.mean(), stated["call_fraction"])
    subset = stated["actual_subset_audit"]
    assert subset["calls"] == called.ravel()[:96].sum()
    assert 0 <= subset["candidate_differences"] <= subset["calls"]
    simulated = np.zeros(groups, dtype=float)
    maps = np.random.default_rng(231780)
    for _ in range(32):
        selection = maps.permutation(called.size)[:int(called.sum())]
        active = np.isin(np.arange(called.size), selection).reshape(called.shape)
        simulated += mask(np.where(active[...,None], bundle, donor)) / 32
    assert simulated.sum() == stated["mean_random_complete"]
    fused = members["confidence_blend"]
    for rival, values in (("intact", members["intact"]), ("random_blend", simulated)):
        d = fused.astype(float) - values
        actual = [d.mean(), *np.quantile(d[draws].mean(1), [.025, .975])]
        saved = stated["paired"][rival]
        np.testing.assert_allclose(actual, [saved["mean"], *saved["interval"]])
    assert (fused & ~members["intact"]).sum() == stated["recovered"]
    assert (~fused & members["intact"]).sum() == stated["lost"]
neutral, decoys = (report["layouts"][k] for k in ("neutral", "decoys"))
d = decoys["paired"]
passed = (d["intact"]["mean"] >= .03 and d["intact"]["interval"][0] > 0
          and d["random_blend"]["mean"] >= .02 and d["random_blend"]["interval"][0] > 0
          and neutral["paired"]["intact"]["interval"][0] >= -.02
          and max(k["call_fraction"] for k in (neutral, decoys)) <= .33
          and all(k["actual_subset_audit"]["candidate_differences"] == 0 for k in (neutral, decoys)))
assert report["decision"] == ("sparse_fusion_adds_repairs" if passed else "unresolved")
print(json.dumps(report["layouts"], indent=2))
