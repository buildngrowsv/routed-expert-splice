"""Audit overwrite-only counterfactuals and the three paired interference contrasts."""
import ast
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
RANKS = runpy.run_path(str(HERE.parent / "numeric_select_72b_study/verify.py"))["check_ranks"]
MODES = ("donor", "orthogonal", "whole")


def verify_panel(rows):
    programs = {}
    for row in rows:
        item = programs.setdefault(row["id"], {})
        item[row["old_variant"], row["chosen"]] = row
        assert row["site"] == [i for i, (a, b) in enumerate(zip(*row["ids"], strict=True)) if a != b][0]
        assert sum(a != b for a, b in zip(*row["ids"], strict=True)) == 1
        compiled = []
        for p, answer in zip(row["prompts"], row["answers"], strict=True):
            code = p + answer + "\n"
            fn = ast.parse(code).body[0]
            original, binding, *filler, overwrite, read = fn.body
            assert original.targets[0].id == overwrite.targets[0].id == binding.value.id
            assert read.value.left.id == (binding.targets[0].id, overwrite.targets[0].id)[row["chosen"]]
            assert 3 <= len(filler) <= 5
            assert int(answer) == int(ast.literal_eval((original, overwrite)[row["chosen"]].value)
                                      + (1 if isinstance(read.value.op, ast.Add) else -1)
                                      * ast.literal_eval(read.value.right))
            exec(compile(code, "<archived crossed overwrite>", "exec"), {})
            compiled.append((ast.literal_eval(original.value), ast.literal_eval(overwrite.value)))
        assert compiled[0][0] == compiled[1][0] and [v[1] for v in compiled] == row["overwrite"]
    for group in programs.values():
        assert list(group) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i in (0, 1):
            saved, current = group[i, 0], group[i, 1]
            assert saved["choices"][0] == saved["choices"][1]
            assert saved["alternate_overwrite_choices"] == current["choices"][::-1]
            assert current["alternate_overwrite_choices"] == current["choices"][::-1]
            assert saved["ids"][0][:saved["site"]] == current["ids"][0][:current["site"]]
    return set(programs)


def qualifies(ranks):
    return (ranks[..., 0] == 1).reshape(-1, 8).all(axis=1)


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    result = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-context-matched-overwrite-72b", "b5a8e086a68aeac4808d9bc14d9c79a722d0ac6c", "done", 0)
    screen = json.loads((R / "screen.json").read_text())
    assert len(screen) == 256 and len(verify_panel(screen)) == 64
    with np.load(R / "screen.npz") as raw:
        RANKS(screen, raw["native"])
        assert int(qualifies(raw["native"]).sum()) == result["screen_eligible"] >= 40
    for field, name in (("map_digest", "numeric_absolute_study/results/map.npz"),
                        ("alignment_digest", "numeric_width_study/72b/alignment.npz")):
        assert sha256((HERE.parent / name).read_bytes()).hexdigest() == result[field]
    panel = json.loads((R / "programs.json").read_text())
    rows = panel["host"]
    assert len(rows) == 1024 and len(verify_panel(rows)) == 256
    assert set(r["id"] for r in rows).isdisjoint(r["id"] for r in screen)
    for donor, host in zip(panel["donor"], rows[::4], strict=True):
        assert donor["source_values"] == host["overwrite"]
        for prompt, answer in zip(donor["prompts"], donor["answers"], strict=True):
            exec(compile(prompt + answer + "\n", "<archived donor>", "exec"), {})
    with np.load(R / "overwrite.npz") as raw:
        ranks = {m: raw[m] for m in ("native", *MODES)}
        dose = raw["norms"]
    assert dose.shape == (2048, 2) and np.isfinite(dose).all()
    np.testing.assert_allclose(dose[:, 0], dose[:, 1], rtol=1e-5, atol=1e-6)
    for v in ranks.values():
        RANKS(rows, v)
    keep = qualifies(ranks["native"])
    n = int(keep.sum())
    assert n == result["eligible"]
    swapped = np.array([r["alternate_overwrite_choices"] for r in rows]).reshape(256, 2, 2, 2)[keep, :, 0]
    current, saved, contamination = {}, {}, {}
    for mode in MODES:
        scores = ranks[mode].reshape(256, 2, 2, 2, 4)[keep]
        current[mode] = int((scores[:, :, 1, :, 1] == 1).all(axis=(1, 2)).sum())
        preserved = scores[:, :, 0, :, 0] == 1
        saved[mode] = int(preserved.all(axis=(1, 2)).sum())
        contamination[mode] = ((~preserved) & (scores[:, :, 0, :, 2] == swapped)).any(axis=(1, 2))
    assert (current, saved, {m: int(x.sum()) for m, x in contamination.items()}) == (
        result["current_switches"], result["saved_preserved"], result["overwritten_answer"])
    draws = np.random.default_rng(847021).integers(n, size=(5000, n))
    for left, right in (("donor", "whole"), ("donor", "orthogonal"), ("whole", "orthogonal")):
        delta = contamination[left].astype(int) - contamination[right]
        cited = result["paired"][f"{left}_minus_{right}"]
        lo, hi = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose([delta.mean(), lo, hi], [cited["mean"], *cited["interval"]])
    matched = result["paired"]["donor_minus_whole"]
    lo, hi = matched["interval"]
    label = "unresolved"
    if n >= 100 and min(current[m] for m in ("donor", "whole")) >= .8*n:
        if matched["mean"] >= .03 and lo > 0:
            label = "donor_leaks_more"
        elif matched["mean"] <= -.03 and hi < 0:
            label = "whole_leaks_more"
        elif lo >= -.03 and hi <= .03 and all(
            result["paired"][k]["interval"][0] > 0 for k in ("donor_minus_orthogonal", "whole_minus_orthogonal")
        ):
            label = "shared_overwrite_interference"
    assert result["decision"] == label
    donor_only = int((contamination["donor"] & ~contamination["whole"]).sum())
    whole_only = int((contamination["whole"] & ~contamination["donor"]).sum())
    discordant = donor_only + whole_only
    lower_tail = sum(comb(discordant, x) for x in range(min(donor_only, whole_only) + 1))
    exact = min(1.0, 2 * lower_tail / 2**discordant)
    print(json.dumps(dict(**result, overwrite_intrusion_donor_only=donor_only,
                          overwrite_intrusion_whole_only=whole_only, posthoc_exact_two_sided=exact,
                          both_edits_intrude=int((contamination["donor"] & contamination["whole"]).sum())),
                     indent=2))


if __name__ == "__main__":
    main()
