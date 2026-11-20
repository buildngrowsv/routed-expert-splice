"""Recheck native routing qualification without running either checkpoint."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE / "receipt.json").read_text())
report = json.loads((HERE / "result.json").read_text())
panel = json.loads((HERE / "screen.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "qwen-mirrored-branch-selector-72b", "19cee73af47266e6e6099d494cc831f5960970f1", "done", 0)
assert len(panel) == 3 * 64
for original, mirrored, inert in zip(panel[::3], panel[1::3], panel[2::3], strict=True):
    assert original["id"] == mirrored["id"] == inert["id"]
    assert original["selector"] == mirrored["selector"] == inert["selector"]
    assert original["answers"] == mirrored["answers"][::-1]
    assert inert["answers"][0] == inert["answers"][1]
    for row in (original, mirrored, inert):
        assert row["site"] == original["site"]
        for prompt, value in zip(row["prompts"], row["answers"], strict=True):
            exec(compile(prompt + value + "\n", "<archived branch prompt>", "exec"), {})
with np.load(HERE / "screen.npz") as raw:
    ranks = raw["native"].reshape(64, 3, 2, 4)
    exact = ranks[..., 0] == 1
    wrong = (~exact[:, :2, :]).sum()
    other_branch = ((ranks[:, :2, :, 1] == 1) & ~exact[:, :2, :]).sum()
    groups = exact.all(axis=(1, 2)).sum()
assert groups == report["screen_eligible"] and groups < 48 and report["decision"] == "benchmark_below_floor"
print(json.dumps(dict(**report, exact_pairs_by_context=exact.all(axis=2).sum(axis=0).tolist(),
                      routing_errors=int(wrong), alternate_branch_errors=int(other_branch)), indent=2))
