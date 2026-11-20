"""Reconstruct the held-out repair ledger and test the preregistered decision."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
receipt = json.loads((RESULTS / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "large-qwen-learned-output-port", "a733c8662949bf34ce8c86f7d4fb17f5a945c014", "done", 0)
report = json.loads((RESULTS / "result.json").read_text())
assert sha256((RESULTS / "ports.safetensors").read_bytes()).hexdigest() == report["port_sha256"]

prompts = json.loads((RESULTS / "programs.json").read_text())
assert {r["prompt"] for r in prompts["train"]}.isdisjoint({r["prompt"] for r in prompts["test"]})
ids = []
for name, start, count in (("train", 80000, 64), ("test", 81000, 256)):
    rows = prompts[name]
    assert len(rows) == 3 * count and len(set(r["prompt"] for r in rows)) == len(rows)
    for g, members in enumerate(zip(rows[::3], rows[1::3], rows[2::3], strict=True)):
        group = f"opener-{start + g}"
        if name == "test":
            ids.append(group)
        assert [m["id"] for m in members] == [f"{group}-{k}" for k in range(3)]
        p = [m["prompt"] for m in members]
        assert [k for k in zip(*p, strict=True) if len(set(k)) > 1] == [("(", "[", "{")]
        for gold, m in enumerate(members):
            head, tail = m["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
            matches = []
            for k, patch in enumerate((")", "]", "}", ":")):
                try:
                    tree = ast.parse(head + patch + tail)
                except SyntaxError:
                    continue
                namespace = {"__builtins__": {"sum": sum}}
                exec(compile(tree, "<verification repair>", "exec"), namespace)
                assert [namespace["transform"](x) for x in [4, -7, 38]] == [4, -7, 38]
                matches.append(k)
            assert [gold] == matches == [m["gold"]]
with np.load(RESULTS / "scores.npz") as raw:
    assert set(raw) == {"host_feature", "donor", "shuffled_donor", "off", "swap", "intact_donor", "native"}
    assert all(v.shape == (256, 3, 4) and np.isfinite(v).all() for v in raw.values())
    labels = {k: v.argmax(-1) for k, v in raw.items()}
complete = {k: (x == [0, 1, 2]).all(axis=-1) for k, x in labels.items()}
assert {k: int(v.sum()) for k, v in complete.items()} == report["complete"]
native = complete["native"]
donor_only = complete["intact_donor"] & ~native
strict = complete["donor"] & ~native & ~complete["shuffled_donor"] & ~complete["host_feature"] & ~complete["off"]
follows = (labels["swap"] == [1, 2, 0]).all(axis=-1)
for k, eligible in (("donor_only_opportunities", donor_only), ("attributed_new_repairs", strict),
                    ("attributed_on_intact_donor_opportunities", strict & donor_only),
                    ("new_repairs_following_cyclic_donor", strict & follows)):
    assert [key for key, keep in zip(ids, eligible, strict=True) if keep] == report[k]
draws = np.random.default_rng(283791).integers(256, size=(5000, 256))
for k, saved in report["donor_minus"].items():
    delta = complete["donor"].astype(int) - complete[k]
    np.testing.assert_allclose(delta.mean(), saved["mean"])
    np.testing.assert_allclose(np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]), saved["interval"])
passed = (donor_only.sum() >= 25 and (strict & donor_only).sum() >= 10
          and all(x["mean"] >= .05 and x["interval"][0] > 0 for x in report["donor_minus"].values()))
assert report["decision"] == ("learned_port_transfers_repairs" if passed else "unresolved")
history = json.loads((RESULTS / "training.json").read_text())
assert set(history) == {"donor", "host_feature", "shuffled_donor"}
assert all(len(v) == 64 and np.isfinite(v).all() for v in history.values())
assert report["trainable_parameters"] == 33 * 5120
raw_margin = int((complete["donor"] & ~complete["off"]).sum())
off_rescued = int((complete["off"] & ~complete["donor"]).sum())
print(json.dumps(dict(complete=report["complete"], donor_only=int(donor_only.sum()), strict=int(strict.sum()),
                      strict_on_intact_donor=int((strict & donor_only).sum()),
                      complete_lost_when_off=raw_margin, complete_rescued_when_off=off_rescued,
                      complete_cyclic_follows=int(follows.sum()),
                      strict_follows=int((strict & follows).sum())), indent=2))
