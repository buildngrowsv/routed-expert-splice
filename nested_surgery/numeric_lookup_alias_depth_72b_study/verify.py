"""Verify the three cuts, matched source panels and registered decision."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE / "results"
base = HERE.parent / "numeric_lookup_alias_72b_study/results"
relocated = HERE.parent / "numeric_lookup_fresh_72b_study/results"
helpers = runpy.run_path(str(HERE.parent / "numeric_lookup_72b_study/verify.py"))
receipt = json.loads((P / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "qwen-dictionary-alias-depth-72b", "346a3c0401050b0078e2a4a21e346e37dd851a09", "done", 0)
report = json.loads((P / "result.json").read_text())
assert report["cuts"] == [17, 40, 60]
for f, expected in ((base / "programs.json", report["alias_programs_sha256"]),
                    (relocated / "programs.json", report["original_programs_sha256"]),
                    (base / "alias.npz", report["raw_inputs"]["alias"]),
                    (relocated / "late.npz", report["raw_inputs"]["relocated"])):
    assert sha256(f.read_bytes()).hexdigest() == expected
alias_rows = json.loads((base / "programs.json").read_text())
late_rows = json.loads((relocated / "programs.json").read_text())["late"]
assert [(r["id"], r["answers"]) for r in alias_rows] == [(r["id"], r["answers"]) for r in late_rows]

modes = ("source", "alias", "relocated")
with np.load(P / "depth.npz") as a:
    names = {f"{m}_{cut}" for m in modes for cut in report["cuts"]}
    assert set(a) == names | {"native_alias", "native_relocated"}
    for key in names:
        helpers["RANKS"](late_rows if key.startswith("relocated") else alias_rows, a[key])
    with np.load(base / "alias.npz") as old:
        for key, trained in (("native_alias", "native"), ("source_17", "early"), ("alias_17", "alias")):
            np.testing.assert_array_equal(a[key], old[trained])
    with np.load(relocated / "late.npz") as old:
        np.testing.assert_array_equal(a["native_relocated"], old["native"])
        np.testing.assert_array_equal(a["relocated_17"], old["whole"])
    eligible = helpers["outcome"](a["native_alias"]) & helpers["outcome"](a["native_relocated"])
    assert eligible.sum() == report["shared"]
    n = int(eligible.sum())
    switches = {}
    for k in names:
        grid = a[k].reshape(256, 3, 2, 4)[eligible]
        switches[k] = (grid[:, :2, :, 1] == 1).all(axis=(1, 2))
        metrics = dict(switch=int(switches[k].sum()), preserve=int((grid[:, 2, :, 0] == 1).all(axis=1).sum()))
        assert metrics == report["metrics"][k]
if n >= 100:
    indices = np.random.default_rng(176614).integers(n, size=(8000, n))
    active, capable = [], []
    for c in (40, 60):
        a, r = f"alias_{c}", f"relocated_{c}"
        observed = report["contrasts"][str(c)]
        for label, baseline in (("alias_minus_initial", "alias_17"), ("alias_minus_relocated", r)):
            raw = switches[a].astype(int) - switches[baseline]
            np.testing.assert_allclose(raw.mean(), observed[label]["mean"])
            bounds = np.percentile(raw[indices].mean(axis=1), [1.25, 98.75])
            np.testing.assert_allclose(bounds, observed[label]["interval"])
        m = report["metrics"]
        if m[r]["switch"] >= .8*n and m[r]["preserve"] >= .95*n:
            active.append(c)
            gain = observed["alias_minus_initial"]
            if m[a]["switch"] >= .5*n and m[a]["preserve"] >= .95*n and gain["mean"] >= .2 and gain["interval"][0] > 0:
                capable.append(c)
    assert active == report["viable_cuts"] and capable == report["recovered_cuts"]
    persistent = active and all(report["contrasts"][str(c)]["alias_minus_relocated"]["interval"][1] < -.2
                                for c in active)
    expected = "alias_recovers_with_depth" if capable else (
        "alias_gap_persists_at_viable_cuts" if persistent else "unresolved")
else:
    expected = "insufficient_shared_coverage"
assert expected == report["decision"]
print(json.dumps({k: report[k] for k in ("decision", "shared", "metrics", "contrasts", "viable_cuts", "recovered_cuts")
                  if k in report}, indent=2))
