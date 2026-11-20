"""Verify the sole source rewrite and rebuild the paired rank comparison."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE / "results"
helpers = runpy.run_path(str(HERE.parent / "numeric_lookup_72b_study/verify.py"))
receipt = json.loads((P / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "qwen-dictionary-late-alias-site-72b", "a2e74ee1055c172ae6a1e02903da7610df6fb37a", "done", 0)
report = json.loads((P / "result.json").read_text())
raw = (HERE.parent / "numeric_lookup_fresh_72b_study/results/programs.json").read_bytes()
assert sha256(raw).hexdigest() == report["source_programs_digest"]
rows = json.loads((P / "programs.json").read_text())
for base, alias in zip(json.loads(raw)["early"], rows, strict=True):
    assert base["id"] == alias["id"] and base["answers"] == alias["answers"]
    assert base["site"] == alias["site"] < alias["alias_site"]
    for before, after, value in zip(base["prompts"], alias["prompts"], alias["answers"], strict=True):
        lines = before.splitlines(keepends=True)
        key = lines[1].split("=")[0].strip()
        insertion = "    selected = " + key + "\n"
        assert after == "".join(lines[:3] + [insertion, lines[3].replace(f"table[{key}]", "table[selected]")] + lines[4:])
        exec(compile(after + value + "\n", "<alias replay>", "exec"), {})
    assert [i for i, (a, b) in enumerate(zip(*alias["ids"], strict=True)) if a != b] == [base["site"]]
with np.load(P / "alias.npz") as archive:
    for array in archive.values():
        helpers["RANKS"](rows, array)
    eligible = helpers["outcome"](archive["native"])
    assert int(eligible.sum()) == report["eligible"] == 183
    switches = {}
    for arm in ("early", "alias"):
        ranks = archive[arm].reshape(256, 3, 2, 4)[eligible]
        switches[arm] = (ranks[:, :2, :, 1] == 1).all(axis=(1, 2))
        assert switches[arm].sum() == report["switches"][arm]
        assert (ranks[:, 2, :, 0] == 1).all(axis=1).sum() == report["preserved"][arm]
    assert helpers["paired"](switches["alias"], switches["early"], 951761) == report["alias_minus_early"]
assert report["decision"] == "unresolved"
print("Verified executable alias rewrite, unchanged source tokens and 183 paired rank outcomes.")
