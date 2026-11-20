"""Recompute the frozen graft's reported effects and check the matched source rewrite."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

here = Path(__file__).resolve().parent
root = here / "results"
receipt = json.loads((root / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "large-qwen-frozen-graft-decoys", "3b8de95dcafd35d53694d9ed53a88183fd746304", "done", 0)
result = json.loads((root / "result.json").read_text())
ports = here.parent / "learned_output_port_study/results/ports.safetensors"
assert sha256(ports.read_bytes()).hexdigest() == result["port_sha256"]
rows = json.loads((root / "programs.json").read_text())
pattern = re.compile(r"'([()\[\]{}]{6})'\)\[0\]  # ([()\[\]{}]{6})")
decoy_code = str.maketrans("()[]{}", "abcdef")
for control, changed in zip(rows["neutral"], rows["decoys"], strict=True):
    assert control["id"] == changed["id"] and control["gold"] == changed["gold"]
    expected, count = pattern.subn(lambda m: "'" + m[1].translate(decoy_code) + "')[0]  # " + m[2].translate(decoy_code), changed["prompt"])
    assert count == 1 and expected == control["prompt"]
    for sample in (control, changed):
        before, after = sample["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        parses = []
        for i, patch in enumerate((")", "]", "}", ":")):
            try:
                tree = ast.parse(before + patch + after)
            except SyntaxError:
                continue
            scope = {"__builtins__": {"sum": sum}}
            exec(compile(tree, "<decoy audit>", "exec"), scope)
            assert scope["transform"](13) == 13
            parses.append(i)
        assert parses == [sample["gold"]]
group_ids = [f"opener-{i}" for i in range(83000, 83256)]
assert [r["id"] for r in rows["decoys"]] == [f"{i}-{j}" for i in group_ids for j in range(3)]
draws = np.random.default_rng(283791).integers(256, size=(5000, 256))
gold = np.array([0, 1, 2])
outcomes = {}
for layout in ("neutral", "decoys"):
    with np.load(root / f"{layout}.npz") as z:
        assert all(x.shape == (256, 3, 4) and np.isfinite(x).all() for x in z.values())
        labels = {k: x.argmax(-1) for k, x in z.items()}
    correct = {k: (x == gold).all(axis=1) for k, x in labels.items()}
    saved = result["endpoints"][layout]
    assert saved["complete"] == {k: int(x.sum()) for k, x in correct.items()}
    initial_lead = correct["intact_donor"] & ~correct["native"]
    strict = correct["donor"] & ~(correct["native"] | correct["host_feature"] | correct["shuffled_donor"] | correct["off"])
    follows = (labels["swap"] == [1, 2, 0]).all(axis=1)
    for k, eligible in (("donor_only_opportunities", initial_lead), ("attributed_new_repairs", strict),
                        ("attributed_on_intact_donor_opportunities", strict & initial_lead),
                        ("new_repairs_following_cyclic_donor", strict & follows)):
        assert saved[k] == [k for k, keep in zip(group_ids, eligible, strict=True) if keep]
    for k, observed in saved["donor_minus"].items():
        delta = correct["donor"].astype(int) - correct[k]
        np.testing.assert_allclose(delta.mean(), observed["mean"])
        np.testing.assert_allclose(np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]), observed["interval"])
    transfer = (initial_lead.sum() >= 25 and (strict & initial_lead).sum() >= 10
                and all(c["mean"] >= .05 and c["interval"][0] > 0 for c in saved["donor_minus"].values()))
    assert (saved["decision"] == "learned_port_transfers_repairs") == transfer
    outcomes[layout] = dict(complete=saved["complete"], strict=int((strict & initial_lead).sum()),
                            donor_off_loss=int((correct["donor"] & ~correct["off"]).sum()),
                            complete_follows=int(follows.sum()), passes=bool(transfer))
gap = outcomes["neutral"]["complete"]["host_feature"] - outcomes["decoys"]["complete"]["host_feature"]
assert gap == result["host_groups_lost"] >= 26 and outcomes["decoys"]["passes"]
assert result["decision"] == "frozen_donor_adds_decoy_repairs"
print(json.dumps(outcomes, indent=2))
