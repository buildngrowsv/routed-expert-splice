"""Reconstruct crossed bracket interventions directly from the public archive."""
import ast
from itertools import combinations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results"


def main():
    rows = json.loads((DATA / "programs.json").read_text())
    former = json.loads((HERE.parent / "large_graft_study/results/programs.json").read_text())
    assert len(set(row["prompt"] for row in rows)) == 768
    assert {r["prompt"] for r in rows}.isdisjoint(r["prompt"] for r in former)
    for g in range(256):
        triplet = rows[3*g:3*g+3]
        prompts = [r["prompt"] for r in triplet]
        assert [chars for chars in zip(*prompts, strict=True) if len(set(chars))>1] == [("(", "[", "{")]
        for j, row in enumerate(triplet):
            assert row["id"] == f"opener-{40000+g}-{j}" and row["gold"] == j
            before, after = row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
            valid = []
            for k, patch in enumerate((")", "]", "}", ":")):
                try:
                    ast.parse(before+patch+after)
                except SyntaxError:
                    continue
                valid.append(k)
            assert valid == [j]
    expected = json.loads((DATA / "result.json").read_text())
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["exit"] == 0 and receipt["revision"] == "4fce8c0bad0a1c6c20436e748549361fe04864c6"
    effect, groups, pair_totals = {}, {}, {}
    with np.load(DATA / "crossed-scores.npz") as archive:
        np.testing.assert_array_equal(archive["gold"], np.tile(np.arange(3),256))
        for mode in ("native","mapped"):
            x = archive[mode]
            assert x.shape == (256,3,3,4) and np.isfinite(x).all()
            choices = np.asarray([x[:,j,j].argmax(1) for j in range(3)]).T
            groups[mode] = int((choices==np.arange(3)).all(1).sum())
            pairs = np.column_stack([(x[:,:,a,a]-x[:,:,a,b]-x[:,:,b,a]+x[:,:,b,b]).mean(1)
                                     for a,b in combinations(range(3),2)])
            pair_totals[mode] = pairs.mean(0).tolist()
            effect[mode] = pairs.mean(1)
        assert archive["off"].shape == (768,4) and np.isfinite(archive["off"]).all()
        groups["off"] = int((archive["off"].argmax(1).reshape(-1,3)==np.arange(3)).all(1).sum())
    boot = np.random.default_rng(445638).choice(256, (5000,256))
    outcome = {}
    for k,v in (*effect.items(), ("mapped_minus_half_native",effect["mapped"]-.5*effect["native"])):
        lo, hi = np.quantile(v[boot].mean(1), [.025,.975])
        outcome[k] = {"mean":float(v.mean()),"interval":[float(lo),float(hi)]}
        np.testing.assert_allclose([v.mean(),lo,hi],[expected[k]["mean"],*expected[k]["interval"]],atol=2e-6)
    assert groups == expected["complete"]
    for k in pair_totals:
        np.testing.assert_allclose(pair_totals[k],expected["pair_means"][k],atol=2e-6)
    outcome["pair_means"] = pair_totals
    outcome["complete"] = groups
    result = "unresolved"
    if outcome["native"]["interval"][0] > 0:
        low, high = outcome["mapped_minus_half_native"]["interval"]
        result = "retained" if low > 0 else "attenuated" if high < 0 else "unresolved"
    assert result == expected["decision"]
    outcome["decision"] = result
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
