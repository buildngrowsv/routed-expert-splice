"""Verify saved-binding semantics, full-vocabulary scores and the preregistered decision."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
RANKS = runpy.run_path(str(HERE.parent / "numeric_select_72b_study/verify.py"))["check_ranks"]
MODES = ("selected", "unused", "orthogonal", "whole")


def verify_panel(rows):
    for saved, current in zip(rows[::2], rows[1::2], strict=True):
        assert saved["id"] == current["id"] and saved["sources"] == current["sources"]
        assert [saved["chosen"], current["chosen"]] == [0, 1]
        for row in (saved, current):
            role = row["chosen"]
            assert row["sites"] == [i for i, (x, y) in enumerate(zip(*row["ids"], strict=True)) if x != y]
            assert row["site"] == row["sites"][role]
            for j, (prompt, answer) in enumerate(zip(row["prompts"], row["answers"], strict=True)):
                written = [int(v) for v in re.findall(r"^    left_\d+ = (\d+)$", prompt, re.M)]
                assert written == [pair[j] for pair in row["sources"]]
                code = prompt + answer + "\n"
                fn = ast.parse(code).body[0]
                name = f"left_{row['id']}"
                copy = f"right_{row['id']}"
                original, binding, *filler, overwrite, read = fn.body
                assert original.targets[0].id == overwrite.targets[0].id == name
                assert binding.targets[0].id == copy and binding.value.id == name
                assert 3 <= len(filler) <= 5 and all(stmt.targets[0].id.startswith("temp_") for stmt in filler)
                assert read.value.left.id == (copy, name)[role]
                scope = {}
                exec(compile(code, "<archived saved-binding prompt>", "exec"), scope)
                assert scope["fetch"]() == written[role] + row["offset"] == int(answer)


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    study = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "qwen-saved-value-vs-overwrite-72b", "cf12ac324612d1779b7eeb6e380968474e5a337b", "done", 0)
    assert study["snapshot"] and study["cut"] == 17
    for key, file in (("map_digest", "numeric_absolute_study/results/map.npz"),
                      ("alignment_digest", "numeric_width_study/72b/alignment.npz")):
        assert sha256((HERE.parent / file).read_bytes()).hexdigest() == study[key]
    panel = json.loads((R / "programs.json").read_text())
    host, screen = panel["host"], panel["screen"]
    assert screen == json.loads((R / "screen.json").read_text())
    assert len(host) == 512 and len(screen) == 128
    assert {row["id"] for row in host}.isdisjoint(row["id"] for row in screen)
    verify_panel(host)
    verify_panel(screen)
    for site, label in enumerate(("left", "right")):
        for source, dest in zip(panel["donor"][label], host[::2], strict=True):
            assert source["source_values"] == dest["sources"][site]
            for prompt, expected in zip(source["prompts"], source["answers"], strict=True):
                exec(compile(prompt + expected + "\n", "<archived donor prompt>", "exec"), {})
    with np.load(R / "screen.npz") as z:
        RANKS(screen, z["native"])
        verified = int((z["native"][..., 0] == 1).reshape(64, 4).all(axis=1).sum())
    assert verified == study["screen_eligible"] >= 48
    with np.load(R / "selection.npz") as z:
        ranks = {m: z[m] for m in ("native", *MODES)}
        scale = z["norms"]
    assert scale.shape == (1024, 2) and np.isfinite(scale).all()
    np.testing.assert_allclose(scale[:, 0], scale[:, 1], rtol=1e-5, atol=1e-6)
    ties = {m: RANKS(host, v) for m, v in ranks.items()}
    eligible = (ranks["native"][..., 0] == 1).reshape(256, 4).all(axis=1)
    n = int(eligible.sum())
    assert n == study["eligible"]
    endpoints = {m: (ranks[m][..., int(m != "unused")] == 1).reshape(256, 2, 2)[eligible]
                 for m in MODES}
    success = {m: values.all(axis=(1, 2)) for m, values in endpoints.items()}
    counts = {m: int(values.sum()) for m, values in success.items()}
    assert counts == study["counts"]
    delta = success["selected"].astype(int) - success["orthogonal"]
    rng = np.random.default_rng(597620)
    low, high = np.percentile(delta[rng.integers(n, size=(5000, n))].mean(axis=1), [2.5, 97.5])
    planned = study["selected_minus_orthogonal"]
    np.testing.assert_allclose([delta.mean(), low, high], [planned["mean"], *planned["interval"]])
    donor = counts["selected"] >= .8 * n and counts["whole"] >= .8 * n
    preservation = counts["unused"] >= .95 * n
    passes = n >= 100 and donor and preservation and delta.mean() >= .2 and low > 0
    assert study["decision"] == ("selective_operand_use" if passes else "unresolved")
    by_role = {m: scored.all(axis=2).sum(axis=0).tolist() for m, scored in endpoints.items()}
    digit = {}
    for row in host:
        for label, token in zip(row["labels"], row["choices"], strict=True):
            assert digit.setdefault(label, token) == token
    unused_errors = np.flatnonzero((ranks["unused"][..., 0] != 1).ravel() & np.repeat(eligible, 4))
    wrong_literal = 0
    for i in unused_errors:
        row, variant = divmod(i, 2)
        alternate = host[row]["sources"][1 - host[row]["chosen"]][variant ^ 1] + host[row]["offset"]
        wrong_literal += int(ranks["unused"][row, variant, 2] == digit[alternate])
    print(json.dumps(dict(**study, paired_saved_and_current=by_role, strict_pass=bool(passes),
                          preservation_minimum=int(np.ceil(.95 * n)), tied_top_candidates=ties,
                          unused_miss_responses=int(len(unused_errors)),
                          unused_miss_top_equals_edited_assignment_plus_offset=wrong_literal), indent=2))


if __name__ == "__main__":
    main()
