"""Audit a native-relative projection hold against the exact previous token traces."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
NEAR = HERE.parent
PANEL = NEAR / "numeric_carry_study"
UTIL = runpy.run_path(str(PANEL / "verify.py"))
INDICES = ("native", "both", "random")


def decoded(path, rows, modes):
    with np.load(path) as data:
        return UTIL["decode_hits"](rows, {m: data[m] for m in modes}, data["terminators"])


def main():
    rec = json.loads((ROOT / "result.json").read_text())
    job = json.loads((ROOT / "receipt.json").read_text())
    assert job["job"] == "qwen-relative-fixed-two-operands-32b"
    assert job["revision"] == "b06d2e63f6b11790a22a5e66be66a08387982c51"
    assert job["status"] == "done" and job["exit"] == 0
    assert rec["reference"] == "relative" and rec["hold_blocks"] == 7
    payload = (PANEL / "results/programs.json").read_bytes()
    assert sha256(payload).hexdigest() == rec["prompts_digest"]
    assert sha256((NEAR / "numeric_absolute_study/results/map.npz").read_bytes()).hexdigest() == rec["map_digest"]
    absolute = NEAR / "numeric_hold_32b_study/results"
    single = NEAR / "numeric_carry_32b_study/results"
    for previous in (absolute, single):
        old = json.loads((previous / "result.json").read_text())
        assert all(old[key] == rec[key] for key in ("prompts_digest", "map_digest", "gain"))
    rows = json.loads(payload)["host"]
    UTIL["verify_programs"](rows)
    moved = decoded(ROOT / "held-32b.npz", rows, INDICES)
    clamped = decoded(absolute / "held-32b.npz", rows, INDICES)
    edit = decoded(single / "carry-32b.npz", rows, (*INDICES, "whole"))
    native = edit["native"].all(axis=1)
    eligible = native & moved["native"].all(axis=1) & clamped["native"].all(axis=1)
    n = int(eligible.sum())
    assert n == rec["common_native"]
    by_arm = {"relative": moved["both"], "relative_random": moved["random"],
              "absolute": clamped["both"], "one_shot": edit["both"]}
    hits = {m: predictions[eligible].all(1) for m, predictions in by_arm.items()}
    assert rec["common_counts"] == {m: int(outcomes.sum()) for m, outcomes in hits.items()}
    both = hits["relative"]
    draws = np.random.default_rng(628177).integers(n, size=(5000, n))
    comparisons, overlaps = {}, {}
    for m in ("relative_random", "absolute", "one_shot"):
        diff = both.astype(int) - hits[m]
        ci = np.percentile(diff[draws].mean(1), [2.5, 97.5])
        np.testing.assert_allclose([diff.mean(), *ci], [rec["relative_minus"][m]["mean"],
                                                       *rec["relative_minus"][m]["interval"]])
        comparisons[m] = (diff.mean(), ci[0])
        overlaps[m] = dict(relative_only=int((diff == 1).sum()), other_only=int((diff == -1).sum()))
    passed = (n >= .95 * native.sum() and comparisons["absolute"][0] >= .04 and comparisons["absolute"][1] > 0
              and comparisons["one_shot"][1] >= -.04 and comparisons["relative_random"][0] >= .2
              and comparisons["relative_random"][1] > 0)
    assert rec["decision"] == ("relative_hold_recovers" if passed else "unresolved")
    widths = np.array([[len(s) for s in row["answers"]] for row in rows])
    directions = {}
    for name, a, b in (("one_to_two", 1, 2), ("two_to_one", 2, 1)):
        group = eligible[:, None] & (widths == a) & (widths[:, ::-1] == b)
        directions[name] = dict(prompt_count=int(group.sum()), counts={m: int(z[group].sum()) for m, z in by_arm.items()})
    print(json.dumps(dict(**rec, paired_switches=overlaps, shared_directions=directions), indent=2))


if __name__ == "__main__":
    main()
