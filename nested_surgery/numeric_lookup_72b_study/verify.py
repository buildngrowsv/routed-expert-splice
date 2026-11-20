"""Recalculate dictionary-routing outcomes and verify the executable counterfactuals."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE / "results"
RANKS = runpy.run_path(str(HERE.parent / "numeric_select_72b_study/verify.py"))["check_ranks"]
MODES = ("donor", "orthogonal", "whole")


def verify_rows(rows):
    for direct, swapped, flat in zip(rows[::3], rows[1::3], rows[2::3], strict=True):
        assert direct["id"] == swapped["id"] == flat["id"]
        assert direct["selector"] == swapped["selector"] == flat["selector"]
        assert direct["answers"][::-1] == swapped["answers"]
        assert flat["answers"][0] == flat["answers"][1]
        for row in (direct, swapped, flat):
            assert row["site"] == direct["site"]
            assert row["site"] == next(i for i, (x, y) in enumerate(zip(*row["ids"], strict=True)) if x != y)
            assert sum(x != y for x, y in zip(*row["ids"], strict=True)) == 1
            for variant, (prompt, value) in enumerate(zip(row["prompts"], row["answers"], strict=True)):
                upto = row["site"] + 1
                assert row["ids"][variant][:upto] == direct["ids"][variant][:upto]
                code = prompt + value + "\n"
                assign, table, _ = ast.parse(code).body[0].body
                key = ast.literal_eval(assign.value)
                mapping = ast.literal_eval(table.value)
                assert key == row["selector"][variant]
                assert sorted(mapping) == sorted(row["selector"]) and list(mapping.values()) == row["payloads"]
                assert mapping[key] + row["offset"] == int(value)
                exec(compile(code, "<archived lookup>", "exec"), {})


def outcome(scores):
    return (scores[..., 0] == 1).reshape(-1, 6).all(axis=1)


def paired(a, b, seed):
    diff = a.astype(int) - b.astype(int)
    n = len(diff)
    rng = np.random.default_rng(seed)
    resampled = diff[rng.integers(n, size=(5000, n))].mean(axis=1)
    return dict(mean=float(diff.mean()), interval=np.percentile(resampled, [2.5, 97.5]).tolist())


def main():
    receipt = json.loads((P / "receipt.json").read_text())
    result = json.loads((P / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-indexed-address-72b", "ddcca579733367d0282894afd3dc752b47a74bc0", "done", 0)
    assert result["lookup"]
    for name, file in (("map_digest", HERE.parent / "numeric_absolute_study/results/map.npz"),
                       ("alignment_digest", HERE.parent / "numeric_width_study/72b/alignment.npz"),
                       ("programs_digest", P / "programs.json")):
        assert sha256(file.read_bytes()).hexdigest() == result[name]
    panel = json.loads((P / "programs.json").read_text())
    host = panel["host"]
    screen = json.loads((P / "screen.json").read_text())
    assert len(host) == 768 and len(screen) == 192 and len(panel["donor"]) == 256
    assert {r["id"] for r in host}.isdisjoint(r["id"] for r in screen)
    verify_rows(screen)
    verify_rows(host)
    for source, row in zip(panel["donor"], host[::3], strict=True):
        assert source["source_values"] == row["selector"]
        for prompt, value in zip(source["prompts"], source["answers"], strict=True):
            exec(compile(prompt + value + "\n", "<archived donor>", "exec"), {})
    with np.load(P / "screen.npz") as z:
        RANKS(screen, z["native"])
        assert int(outcome(z["native"]).sum()) == result["screen_eligible"] >= 48
    with np.load(P / "selector.npz") as z:
        scores = {mode: z[mode] for mode in ("native", *MODES)}
        dose = z["norms"]
    assert dose.shape == (1536, 2) and np.isfinite(dose).all()
    np.testing.assert_allclose(dose[:, 0], dose[:, 1], atol=1e-6, rtol=1e-5)
    for array in scores.values():
        RANKS(host, array)
    eligible = outcome(scores["native"])
    n = int(eligible.sum())
    assert n == result["eligible"]
    groups, switch_both, joint = {}, {}, {}
    for mode in MODES:
        columns = scores[mode].reshape(256, 3, 2, 4)[eligible]
        routing, mirrored = (columns[:, :2, :, 1] == 1).all(axis=2).T
        inert = (columns[:, 2, :, 0] == 1).all(axis=1)
        switch_both[mode] = routing & mirrored
        joint[mode] = switch_both[mode] & inert
        groups[mode] = dict(routing=int(routing.sum()), mirrored=int(mirrored.sum()),
                            both_switch=int(switch_both[mode].sum()), inert=int(inert.sum()),
                            joint=int(joint[mode].sum()))
    assert groups == result["counts"]
    contrast = paired(joint["donor"], joint["orthogonal"], 597621)
    np.testing.assert_allclose(contrast["mean"], result["donor_minus_orthogonal"]["mean"])
    np.testing.assert_allclose(contrast["interval"], result["donor_minus_orthogonal"]["interval"])
    passed = (n >= 100 and min(groups[k]["both_switch"] for k in ("donor", "whole")) >= .8*n
              and groups["donor"]["inert"] >= .95*n and contrast["mean"] >= .2 and contrast["interval"][0] > 0)
    assert result["decision"] == ("selector_transfer" if passed else "unresolved")
    print(json.dumps(dict(**result, minimum_switches=int(np.ceil(.8*n)),
                          both_switch_donor_minus_whole=paired(switch_both["donor"], switch_both["whole"], 193617),
                          switch_overlap=int((switch_both["donor"] & switch_both["whole"]).sum())), indent=2))


if __name__ == "__main__":
    main()
