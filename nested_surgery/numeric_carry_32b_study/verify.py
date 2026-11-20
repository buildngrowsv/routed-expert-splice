"""Verify the 32B carry run with Python execution and archived token scores."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVED = HERE.parent / "numeric_carry_study"
PREVIOUS = runpy.run_path(str(ARCHIVED / "verify.py"))
MODES = ("native", "both", "random", "whole")
decode = PREVIOUS["decode_hits"]


def main():
    root = HERE / "results"
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["job"] == "qwen-borrowed-two-operands-32b"
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "f0fc062c821766701472077421ec9cc02c6d11e6"
    report = json.loads((root / "result.json").read_text())
    scalar = json.loads((HERE.parent / "numeric_scalar_study/audit.json").read_text())
    assert report["gain"] == scalar["gain"]
    screen = json.loads((ARCHIVED / "results/screen.json").read_text())
    PREVIOUS["verify_programs"](screen)
    with np.load(root / "screen.npz") as z:
        first = decode(screen, {"native": z["native"]}, z["terminators"])["native"]
        assert int(first.all(1).sum()) == report["screen_eligible"]
    if report["screen_eligible"] < 64:
        assert report["decision"] == "benchmark_below_floor"
        print(json.dumps(report, indent=2))
        return
    payload = (ARCHIVED / "results/programs.json").read_bytes()
    assert sha256(payload).hexdigest() == report["prompts_digest"]
    rows = json.loads(payload)["host"]
    PREVIOUS["verify_programs"](rows)
    digest = sha256((HERE.parent / "numeric_absolute_study/results/map.npz").read_bytes()).hexdigest()
    assert digest == report["map_digest"] == scalar["map_digest"]
    with np.load(root / "carry-32b.npz") as z:
        left, right = z["both_norm"], z["random_norm"]
        assert left.shape == right.shape == (len(rows) * 4, 2)
        assert np.isfinite(left).all() and np.isfinite(right).all() and (left > 0).all()
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
        hits = decode(rows, {m: z[m] for m in MODES}, z["terminators"])
    eligible = hits["native"].all(1)
    mask = eligible[:, None]
    n = int(eligible.sum())
    correct = {m: int(h[eligible].all(1).sum()) for m, h in hits.items()}
    assert report["eligible"] == n and report["correct"] == correct
    widths = np.array([[len(a) for a in row["answers"]] for row in rows])
    for name, source, target in (("one_to_two", 1, 2), ("two_to_one", 2, 1)):
        selected = mask & (widths == source) & (widths[:, ::-1] == target)
        checks = {m: int(h[selected].sum()) for m, h in hits.items() if m != "native"}
        assert report["direction_counts"][name] == dict(prompts=int(selected.sum()), correct=checks)
    if n:
        change = hits["both"][eligible].all(1).astype(int) - hits["random"][eligible].all(1)
        rows = np.random.default_rng(814225).integers(n, size=(5000, n))
        low, high = np.percentile(change[rows].mean(1), [2.5, 97.5])
        assert report["both_minus_random"] == dict(mean=change.mean(), interval=[low, high])
    else:
        low = 0.
    passed = n >= 100 and min(correct["both"], correct["whole"]) >= .8 * n and change.mean() >= .2 and low > 0
    assert report["decision"] == ("carry_composes" if passed else "unresolved")
    borrowed = hits["both"][eligible].all(1)
    whole = hits["whole"][eligible].all(1)
    overlap = dict(both_correct=int((borrowed & whole).sum()),
                   borrowed_only=int((borrowed & ~whole).sum()),
                   whole_only=int((~borrowed & whole).sum()),
                   neither=int((~borrowed & ~whole).sum()))
    print(json.dumps(dict(**report, overlap=overlap), indent=2))


if __name__ == "__main__":
    main()
