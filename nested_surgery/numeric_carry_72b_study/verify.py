"""Audit transferred carry in 72B against the original programs and the 32B panel."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
ORIGINAL = HERE.parent / "numeric_carry_study" / "results"
PRIOR_32 = HERE.parent / "numeric_carry_32b_study" / "results"
UTIL = runpy.run_path(str(ORIGINAL.parent / "verify.py"))
ARMS = ("native", "both", "random", "whole")


def main():
    job = json.loads((ROOT / "receipt.json").read_text())
    report = json.loads((ROOT / "result.json").read_text())
    assert (job["job"], job["revision"], job["status"], job["exit"]) == (
        "qwen-aligned-two-operands-72b", "e240a141e91e198a3729870d54822cdbef7a0403", "done", 0)
    payload = (ORIGINAL / "programs.json").read_bytes()
    assert sha256(payload).hexdigest() == report["prompts_digest"]
    prefix = HERE.parent
    m = prefix / "numeric_absolute_study/results/map.npz"
    a = prefix / "numeric_width_study/72b/alignment.npz"
    assert sha256(m.read_bytes()).hexdigest() == report["map_digest"]
    assert sha256(a.read_bytes()).hexdigest() == report["alignment_digest"]
    panel = json.loads(payload)
    host = panel["host"]
    screen = json.loads((ORIGINAL / "screen.json").read_text())
    decode = UTIL["decode_hits"]
    for dataset in (host, screen):
        UTIL["verify_programs"](dataset)
    for name, donor_rows in panel["donor"].items():
        for source, dest in zip(donor_rows, host, strict=True):
            assert source["source_values"] == dest[name]
    with np.load(ROOT / "screen.npz") as data:
        native = decode(screen, {"native": data["native"]}, data["terminators"])["native"].all(1)
    assert report["screen_eligible"] == int(native.sum())
    with np.load(ROOT / "carry-72b.npz") as data:
        both, random = data["both_norm"], data["random_norm"]
        assert both.shape == random.shape == (len(host)*4, 2) and np.isfinite(both).all()
        np.testing.assert_allclose(both, random, rtol=1e-5, atol=1e-6)
        new = decode(host, {m: data[m] for m in ARMS}, data["terminators"])
    with np.load(PRIOR_32 / "carry-32b.npz") as data:
        old = decode(host, {m: data[m] for m in ARMS}, data["terminators"])
    valid = new["native"].all(1)
    n = int(valid.sum())
    shared = valid & old["native"].all(1)
    counts = {m: int(z[valid].all(1).sum()) for m, z in new.items()}
    assert report["eligible"] == n and report["correct"] == counts
    width = np.array([[len(x) for x in r["answers"]] for r in host])
    for label, a, b in (("one_to_two", 1, 2), ("two_to_one", 2, 1)):
        selected = valid[:, None] & (width == a) & (width[:, ::-1] == b)
        values = {m: int(row[selected].sum()) for m, row in new.items() if m != "native"}
        assert report["direction_counts"][label] == dict(correct=values, prompts=int(selected.sum()))
    delta = new["both"][valid].all(1).astype(int) - new["random"][valid].all(1)
    sample = np.random.default_rng(814225).integers(n, size=(5000, n))
    low, high = np.percentile(delta[sample].mean(axis=1), [2.5, 97.5])
    np.testing.assert_allclose([delta.mean(), low, high],
                               [report["both_minus_random"]["mean"], *report["both_minus_random"]["interval"]])
    success = n >= 100 and min(counts["both"], counts["whole"]) >= .8*n and delta.mean() >= .2 and low > 0
    assert report["decision"] == ("carry_composes" if success else "unresolved")
    overlap_counts = {size: {m: int(x[shared].all(1).sum()) for m, x in outcomes.items()}
                      for size, outcomes in (("32b", old), ("72b", new))}
    shift = new["both"][shared].all(1).astype(int) - old["both"][shared].all(1)
    sample = np.random.default_rng(553762).integers(len(shift), size=(5000, len(shift)))
    effect = dict(mean=float(shift.mean()), interval=np.percentile(shift[sample].mean(axis=1), [2.5, 97.5]).tolist())
    print(json.dumps(dict(**report, shared_native=int(shared.sum()), shared_counts=overlap_counts,
                          paired_gain_72b=effect), indent=2))


if __name__ == "__main__":
    main()
