"""Recompute the selected and unused assignment outcomes from archived ranks."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
ARMS = ("selected", "unused", "orthogonal", "whole")


def verify_programs(rows):
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        assert left["id"] == right["id"] and left["sources"] == right["sources"]
        assert [left["chosen"], right["chosen"]] == [0, 1]
        for row in (left, right):
            assert row["sites"] == [i for i, pair in enumerate(zip(*row["ids"], strict=True))
                                    if pair[0] != pair[1]]
            assert row["site"] == row["sites"][row["chosen"]]
            for i, (prompt, expected) in enumerate(zip(row["prompts"], row["answers"], strict=True)):
                source = [int(d) for d in re.findall(r"^    (?:left|right)_\d+ = (\d+)$", prompt, re.M)]
                assert source == [values[i] for values in row["sources"]]
                scope = {}
                exec(compile(prompt + expected + "\n", "<archived host prompt>", "exec"), scope)
                assert scope["fetch"]() == source[row["chosen"]] + row["offset"] == int(expected)
                assert int(row["answers"][1 - i]) != int(expected)


def check_ranks(rows, ranks):
    assert ranks.shape == (len(rows), 2, 4) and np.isfinite(ranks).all()
    expected = np.asarray([r["choices"] for r in rows])
    tied_top = 0
    for col, tokens in ((0, expected), (1, expected[:, ::-1])):
        first = ranks[..., col] == 1
        is_argmax = ranks[..., 2] == tokens
        assert (is_argmax | ~first).all()
        tied_top += int((is_argmax & ~first).sum())
    return tied_top


def main():
    receipt = json.loads((ROOT / "receipt.json").read_text())
    report = json.loads((ROOT / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-selected-vs-unused-72b", "1d24543e9b451368dbeefedbb881e55588a6b256", "done", 0)
    for key, file in (("map_digest", "numeric_absolute_study/results/map.npz"),
                      ("alignment_digest", "numeric_width_study/72b/alignment.npz")):
        assert sha256((HERE.parent / file).read_bytes()).hexdigest() == report[key]
    panel = json.loads((ROOT / "programs.json").read_text())
    host, screen = panel["host"], panel["screen"]
    assert len(host) == 512 and len(screen) == 128
    assert {r["id"] for r in host}.isdisjoint(r["id"] for r in screen)
    verify_programs(host)
    verify_programs(screen)
    for role, name in enumerate(("left", "right")):
        for source, dest in zip(panel["donor"][name], host[::2], strict=True):
            assert source["source_values"] == dest["sources"][role]
            for prompt, answer in zip(source["prompts"], source["answers"], strict=True):
                exec(compile(prompt + answer + "\n", "<archived donor prompt>", "exec"), {})
    with np.load(ROOT / "screen.npz") as raw:
        ranks = raw["native"]
        check_ranks(screen, ranks)
        screen_count = int((ranks[:, :, 0] == 1).reshape(64, 4).all(1).sum())
    assert screen_count == report["screen_eligible"] and screen_count >= 48
    with np.load(ROOT / "selection.npz") as raw:
        data = {m: raw[m] for m in ("native", *ARMS)}
        norms = raw["norms"]
    assert norms.shape == (1024, 2) and np.isfinite(norms).all()
    np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
    ties = {mode: check_ranks(host, ranks) for mode, ranks in data.items()}
    eligible = (data["native"][:, :, 0] == 1).reshape(256, 4).all(1)
    count = int(eligible.sum())
    assert count == report["eligible"] and count == report["groups"] - int((~eligible).sum())
    goals = {m: (data[m][:, :, 0 if m == "unused" else 1] == 1).reshape(256, 2, 2)[eligible]
             for m in ARMS}
    all_contexts = {m: valid.all(axis=(1, 2)) for m, valid in goals.items()}
    counts = {m: int(valid.sum()) for m, valid in all_contexts.items()}
    assert counts == report["counts"]
    delta = all_contexts["selected"].astype(int) - all_contexts["orthogonal"]
    samples = np.random.default_rng(597620).integers(count, size=(5000, count))
    lower, upper = np.percentile(delta[samples].mean(axis=1), [2.5, 97.5])
    expected = report["selected_minus_orthogonal"]
    np.testing.assert_allclose([delta.mean(), lower, upper], [expected["mean"], *expected["interval"]])
    passed = (count >= 100 and min(counts["selected"], counts["whole"]) >= .8 * count
              and counts["unused"] >= .95 * count and delta.mean() >= .2 and lower > 0)
    assert report["decision"] == ("selective_operand_use" if passed else "unresolved")
    by_role = {m: valid.all(axis=2).sum(axis=0).tolist() for m, valid in goals.items()}
    print(json.dumps(dict(**report, paired_left_and_right=by_role, tied_top_candidates=ties), indent=2))


if __name__ == "__main__":
    main()
