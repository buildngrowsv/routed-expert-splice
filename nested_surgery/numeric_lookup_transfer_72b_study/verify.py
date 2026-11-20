"""Audit the matched early- and late-key donor interventions."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
EARLY = HERE.parent / "numeric_lookup_72b_study"
LATE = HERE.parent / "numeric_lookup_order_72b_study"
CHECKS = runpy.run_path(str(EARLY / "verify.py"))


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    report = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-late-key-donor-transfer-72b", "436d2c7edfa8b3ab6f103cd22678f61d1d5633b7", "done", 0)
    for field, path in (("map_digest", HERE.parent / "numeric_absolute_study/results/map.npz"),
                        ("alignment_digest", HERE.parent / "numeric_width_study/72b/alignment.npz"),
                        ("original_programs_digest", EARLY / "results/programs.json")):
        assert sha256(path.read_bytes()).hexdigest() == report[field]
    original = json.loads((EARLY / "results/programs.json").read_text())
    rows = json.loads((LATE / "results/programs.json").read_text())
    for old, new in zip(original["host"], rows, strict=True):
        assert old["answers"] == new["answers"] and new["site"] > old["site"]
        for p, q, answer in zip(old["prompts"], new["prompts"], old["answers"], strict=True):
            a = p.splitlines(keepends=True)
            a[1], a[2] = a[2], a[1]
            assert "".join(a) == q
            exec(compile(q + answer + "\n", "<archived late key>", "exec"), {})
    with np.load(EARLY / "results/selector.npz") as old, np.load(R / "late-donor.npz") as new:
        early = {k: old[k] for k in ("native", "donor")}
        late = {k: new[k] for k in ("native", "donor", "whole", "orthogonal")}
        doses = new["norms"]
    assert doses.shape == (1536, 2) and np.isfinite(doses).all()
    np.testing.assert_allclose(doses[:, 0], doses[:, 1], rtol=1e-5, atol=1e-6)
    with np.load(LATE / "results/moved-key.npz") as previous:
        for k in ("native", "whole"):
            np.testing.assert_array_equal(late[k][..., :3], previous[k][..., :3])
    for points in late.values():
        CHECKS["RANKS"](rows, points)
    eligible = CHECKS["outcome"](early["native"]) & CHECKS["outcome"](late["native"])
    n = int(eligible.sum())
    assert n == report["eligible"]
    switches, preserved = {}, {}
    modes = dict(early_donor=early["donor"], late_donor=late["donor"],
                 late_orthogonal=late["orthogonal"], late_whole=late["whole"])
    for m, a in modes.items():
        sel = a.reshape(256, 3, 2, 4)[eligible]
        switches[m] = (sel[:, :2, :, 1] == 1).all(axis=(1, 2))
        preserved[m] = int((sel[:, 2, :, 0] == 1).all(axis=1).sum())
    counts = {k: int(v.sum()) for k, v in switches.items()}
    assert counts == report["mirrored_switches"] and preserved == report["constant_preserved"]
    for mode in ("early_donor", "late_orthogonal", "late_whole"):
        p = CHECKS["paired"](switches["late_donor"], switches[mode], 876211)
        stated = report["late_donor_minus"][mode]
        np.testing.assert_allclose([p["mean"], *p["interval"]], [stated["mean"], *stated["interval"]])
    order = report["late_donor_minus"]["early_donor"]
    ctrl = report["late_donor_minus"]["late_orthogonal"]
    yes = (n >= 100 and min(counts[k] for k in ("late_donor", "late_whole")) >= .8*n
           and preserved["late_donor"] >= .95*n and order["mean"] >= .15 and order["interval"][0] > 0
           and ctrl["mean"] >= .2 and ctrl["interval"][0] > 0)
    assert report["decision"] == ("late_key_donor_transfer" if yes else "unresolved")
    print(json.dumps(dict(**report,
                          late_only_donor=int((switches["late_donor"] & ~switches["early_donor"]).sum()),
                          early_only_donor=int((switches["early_donor"] & ~switches["late_donor"]).sum())),
                     indent=2))


if __name__ == "__main__":
    main()
