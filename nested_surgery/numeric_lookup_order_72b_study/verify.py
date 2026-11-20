"""Verify paired dictionary declaration order and recalculate whole-state switching."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
PRIOR = HERE.parent / "numeric_lookup_72b_study"
CHECKS = runpy.run_path(str(PRIOR / "verify.py"))


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    report = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-dictionary-key-order-72b", "2ceb1b12fdc7238065394c273ebe6be2936ed1d8", "done", 0)
    early_raw = (PRIOR / "results/programs.json").read_bytes()
    prior_ranks = PRIOR / "results/selector.npz"
    assert sha256(early_raw).hexdigest() == report["original_programs_digest"]
    assert sha256(prior_ranks.read_bytes()).hexdigest() == report["original_ranks_digest"]
    early = json.loads(early_raw)["host"]
    late = json.loads((R / "programs.json").read_text())
    assert len(early) == len(late) == 768
    CHECKS["verify_rows"](early)
    for left, right in zip(early, late, strict=True):
        assert {k: v for k, v in left.items() if k not in ("prompts", "ids", "site")} == {
            k: v for k, v in right.items() if k not in ("prompts", "ids", "site")}
        for before, after, answer in zip(left["prompts"], right["prompts"], right["answers"], strict=True):
            other = before.splitlines(keepends=True)
            other[1], other[2] = other[2], other[1]
            assert "".join(other) == after
            exec(compile(after + answer + "\n", "<archived moved key>", "exec"), {})
        changed = [i for i, (p, q) in enumerate(zip(*right["ids"], strict=True)) if p != q]
        assert changed == [right["site"]] and right["site"] > left["site"]
    panels = {}
    for name, rows, file in (("early", early, prior_ranks), ("late", late, R / "moved-key.npz")):
        with np.load(file) as z:
            panels[name] = {m: z[m].reshape(256, 3, 2, 4) for m in ("native", "whole")}
            for m in panels[name]:
                CHECKS["RANKS"](rows, z[m])
    native = {k: (scores["native"][..., 0] == 1).all(axis=(1, 2)) for k, scores in panels.items()}
    assert int(native["early"].sum()) == report["native_early"]
    assert int(native["late"].sum()) == report["native_late"]
    shared = native["early"] & native["late"]
    n = int(shared.sum())
    assert n == report["shared_native_groups"]
    counts, switched = {}, {}
    for arm, scores in panels.items():
        rank = scores["whole"][shared]
        switched[arm] = (rank[:, :2, :, 1] == 1).all(axis=(1, 2))
        preserved = (rank[:, 2, :, 0] == 1).all(axis=1)
        counts[arm] = dict(switch_both=int(switched[arm].sum()), inert=int(preserved.sum()),
                           switch_and_preserve=int((switched[arm] & preserved).sum()))
    assert counts == report["whole"]
    effect = CHECKS["paired"](switched["late"], switched["early"], 325140)
    np.testing.assert_allclose(effect["mean"], report["late_minus_early"]["mean"])
    np.testing.assert_allclose(effect["interval"], report["late_minus_early"]["interval"])
    criteria = (n >= 100 and counts["late"]["switch_both"] >= .8*n and counts["late"]["inert"] >= .95*n
                and effect["mean"] >= .15 and effect["interval"][0] > 0)
    assert report["decision"] == ("key_order_restores_whole_state_routing" if criteria else "unresolved")
    print(json.dumps(dict(**report,
                          early_only_native=int((native["early"] & ~native["late"]).sum()),
                          late_only_native=int((native["late"] & ~native["early"]).sum()),
                          early_only_switch=int((switched["early"] & ~switched["late"]).sum()),
                          late_only_switch=int((switched["late"] & ~switched["early"]).sum())), indent=2))


if __name__ == "__main__":
    main()
