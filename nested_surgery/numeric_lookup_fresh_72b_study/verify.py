"""Audit the disjoint early/late dictionary donor comparison."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ORIGINAL = HERE.parent / "numeric_lookup_72b_study"
CONTROL = runpy.run_path(str(ORIGINAL / "verify.py"))


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    reported = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-late-key-donor-fresh-72b", "ad2451d54e4b6c70aa73749686402a0f317a4897", "done", 0)
    for name, path in (("map_digest", HERE.parent / "numeric_absolute_study/results/map.npz"),
                       ("alignment_digest", HERE.parent / "numeric_width_study/72b/alignment.npz"),
                       ("programs_digest", R / "programs.json")):
        assert sha256(path.read_bytes()).hexdigest() == reported[name]
    programs = json.loads((R / "programs.json").read_text())
    first, last = programs["early"], programs["late"]
    prior = json.loads((ORIGINAL / "results/programs.json").read_text())
    assert len(first) == len(last) == 768 and len(programs["donor"]) == 256
    assert {r["id"] for r in first} == set(range(176000,176256))
    assert {r["id"] for r in first}.isdisjoint(r["id"] for r in prior["host"])
    assert {r["id"] for r in programs["donor"]}.isdisjoint(r["id"] for r in prior["donor"])
    CONTROL["verify_rows"](first)
    for a, b in zip(first, last, strict=True):
        assert a["choices"] == b["choices"] and a["answers"] == b["answers"] and a["site"] < b["site"]
        for x, y, answer in zip(a["prompts"], b["prompts"], a["answers"], strict=True):
            lines = x.splitlines(keepends=True)
            lines[1], lines[2] = lines[2], lines[1]
            assert "".join(lines) == y
            exec(compile(y + answer + "\n", "<fresh matched lookup>", "exec"), {})
        assert [i for i, pair in enumerate(zip(*b["ids"], strict=True)) if pair[0] != pair[1]] == [b["site"]]
    for source, target in zip(programs["donor"], first[::3], strict=True):
        assert source["source_values"] == target["selector"]
        for p, answer in zip(source["prompts"], source["answers"], strict=True):
            exec(compile(p + answer + "\n", "<fresh donor declaration>", "exec"), {})
    raw = {}
    native = {}
    for order in ("early", "late"):
        with np.load(R / f"{order}.npz") as f:
            norms = f["norms"]
            assert norms.shape == (1536,2) and np.isfinite(norms).all()
            np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
            raw[order] = {k: f[k] for k in ("native","donor","orthogonal","whole")}
        for data in raw[order].values():
            CONTROL["RANKS"](programs[order], data)
        native[order] = CONTROL["outcome"](raw[order]["native"])
    keep = native["early"] & native["late"]
    n = int(keep.sum())
    assert n == reported["eligible"]
    by_mode, survived = {}, {}
    for side, arm in (("early","donor"),("early","whole"),("early","orthogonal"),
                      ("late","donor"),("late","whole"),("late","orthogonal")):
        rank = raw[side][arm].reshape((256,3,2,4))[keep]
        mode = f"{side}_{arm}"
        by_mode[mode] = (rank[:, :2, :, 1] == 1).all(axis=(1, 2))
        survived[mode] = int((rank[:, 2, :, 0] == 1).all(axis=1).sum())
    summary = {m: int(x.sum()) for m, x in by_mode.items()}
    for k, v in reported["mirrored_switches"].items():
        assert v == summary[k] and reported["constant_preserved"][k] == survived[k]
    for arm in ("early_donor","late_orthogonal","late_whole"):
        est = CONTROL["paired"](by_mode["late_donor"], by_mode[arm], 876211)
        sealed = reported["late_donor_minus"][arm]
        np.testing.assert_allclose([est["mean"], *est["interval"]], [sealed["mean"], *sealed["interval"]])
    a, b = (reported["late_donor_minus"][m] for m in ("early_donor","late_orthogonal"))
    gate = (n >= 100 and min(summary[k] for k in ("late_donor","late_whole")) >= .8*n
            and survived["late_donor"] >= .95*n and a["mean"] >= .15 and a["interval"][0] > 0
            and b["mean"] >= .20 and b["interval"][0] > 0)
    assert reported["decision"] == ("late_key_donor_transfer" if gate else "unresolved")
    print(json.dumps(dict(**reported, all_mode_switches=summary, all_mode_preservation=survived,
                          native_groups={k:int(x.sum()) for k,x in native.items()},
                          late_donor_only_vs_whole=int((by_mode["late_donor"] & ~by_mode["late_whole"]).sum()),
                          late_whole_only_vs_donor=int((by_mode["late_whole"] & ~by_mode["late_donor"]).sum())),
                     indent=2))


if __name__ == "__main__":
    main()
