"""Independently reconstruct the 32B midpoint and shared 72B comparison."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

here = Path(__file__).resolve().parent
root = here / "results"
prior = here.parent / "numeric_lookup_order_32b_study/results"
depth72 = here.parent / "numeric_lookup_alias_depth_72b_study/results/depth.npz"
atlas = runpy.run_path(str(here.parent / "numeric_lookup_72b_study/verify.py"))
receipt = json.loads((root / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "qwen-dictionary-alias-midpoint-32b", "8d6d1a1fd0a0886dfbab2a381b73e4bed6683758", "done", 0)
result = json.loads((root / "result.json").read_text())
assert result["cut"] == 32 and sha256((prior / "programs.json").read_bytes()).hexdigest() == result["source32_sha256"]
rows = json.loads((root / "programs.json").read_text())
history = json.loads((prior / "programs.json").read_text())
assert rows["relocated"] == history["late"]
for base, current in zip(history["early"], rows["alias"], strict=True):
    assert base["site"] == current["site"] < current["alias_site"]
    assert base["id"] == current["id"] and base["answers"] == current["answers"]
    for before, after, expected in zip(base["prompts"], current["prompts"], base["answers"], strict=True):
        lines = before.splitlines(keepends=True)
        key = lines[1].split("=")[0].strip()
        assert after == "".join([*lines[:3], f"    selected = {key}\n",
                                 lines[3].replace(f"table[{key}]", "table[selected]"), *lines[4:]])
        exec(compile(after+expected+"\n", "<archived 32B alias>", "exec"), {})
with np.load(root / "midpoint.npz") as raw:
    for mode, values in raw.items():
        atlas["RANKS"](rows["relocated" if "relocated" in mode else "alias"], values)
    with np.load(prior / "late.npz") as old:
        np.testing.assert_array_equal(raw["native_relocated"][..., :3], old["native"][..., :3])
    eligible32 = atlas["outcome"](raw["native_alias"]) & atlas["outcome"](raw["native_relocated"])
    projected32 = {m: raw[m] for m in ("source", "alias", "relocated")}
with np.load(depth72) as z:
    eligible72 = atlas["outcome"](z["native_alias"]) & atlas["outcome"](z["native_relocated"])
    projected72 = {m: z[m+"_40"] for m in ("alias", "relocated")}
same = eligible32 & eligible72


def summarize(arms, panel, seed, expected):
    combined = {k: v.reshape(256, 3, 2, 4)[panel] for k, v in arms.items()}
    flips = {k: (v[:, :2, :, 1] == 1).all(axis=(1, 2)) for k, v in combined.items()}
    kept = {k: int((v[:, 2, :, 0] == 1).all(axis=1).sum()) for k, v in combined.items()}
    assert expected["shared"] == int(panel.sum())
    assert expected["switched"] == {k: int(v.sum()) for k, v in flips.items()}
    assert expected["constant_preserved"] == kept
    if panel.any():
        got = atlas["paired"](flips["alias"], flips["relocated"], seed)
        assert got == expected["alias_minus_relocated"]


summarize(projected32, eligible32, 491627, result["local"])
summarize(projected32, same, 713511, result["cross_host_shared"]["at_32b"])
summarize(projected72, same, 713511, result["cross_host_shared"]["at_72b"])
local = result["local"]
n = local["shared"]
control = n >= 100 and local["switched"]["relocated"] >= .8*n and local["constant_preserved"]["relocated"] >= .95*n
assert control == result["control_open"]
decision = "unresolved"
if control and local["alias_minus_relocated"]["interval"][1] <= -.20:
    decision = "alias_gap_replicates_at_32b"
elif control and local["switched"]["alias"] >= .5*n and local["constant_preserved"]["alias"] >= .95*n:
    decision = "32b_alias_switches_at_midpoint"
assert decision == result["decision"]
print(json.dumps(dict(decision=decision, local=local, cross_host_shared=result["cross_host_shared"]), indent=2))
