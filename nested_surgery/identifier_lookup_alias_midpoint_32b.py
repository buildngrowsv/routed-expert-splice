"""Repeat the informative midpoint contrast in the independently trained 32B host."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_lookup_alias_72b import alias_rows, evaluate
from identifier_lookup_order_72b import evaluate as score_relocated
from identifier_numeric_carry_32b import HOST_32, verified_result
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files

PLACEMENT = "qwen-dictionary-order-second-host-32b"
PLACEMENT_REV = "4d0348e77fafbb3a0a6c12015f20cc341197288f"
DEPTH = "qwen-dictionary-alias-depth-72b"
DEPTH_REV = "346a3c0401050b0078e2a4a21e346e37dd851a09"
CUT = 32


def metrics(arms, eligible, seed, labels):
    n = int(eligible.sum())
    switched, preservation = {}, {}
    for label in labels:
        rows = arms[label].reshape(-1, 3, 2, 4)[eligible]
        switched[label] = (rows[:, :2, :, 1] == 1).all(axis=(1, 2))
        preservation[label] = int((rows[:, 2, :, 0] == 1).all(axis=1).sum())
    switch_counts = {k: int(v.sum()) for k, v in switched.items()}
    comparison = None
    if n:
        sample = np.random.default_rng(seed).integers(n, size=(5000, n))
        comparison = interval(switched["alias"].astype(int) - switched["relocated"], sample)
    return dict(shared=n, switched=switch_counts, constant_preserved=preservation,
                alias_minus_relocated=comparison)


def main():
    begun = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    out = Path(os.environ["STUDIO_JOB_DIR"])
    registered = verified_result(root / PLACEMENT, PLACEMENT_REV, "late_key_control_replicates_at_32b")
    control = verified_result(root / DEPTH, DEPTH_REV, "alias_gap_persists_at_viable_cuts")
    assert registered["source_programs_digest"] == control["original_programs_sha256"]
    archived_rows = (root / PLACEMENT / "programs.json").read_bytes()
    panel = json.loads(archived_rows)
    path = checkpoint_files(*HOST_32)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    alias = alias_rows(tokenizer, panel["early"])
    moved = panel["late"]
    assert [r["id"] for r in alias] == [r["id"] for r in moved]
    assert [r["answers"] for r in alias] == [r["answers"] for r in moved]
    model, _ = load_model(path, strict=True)
    assert len(model.model.layers) == 2*CUT and model.model.embed_tokens.weight.dtype == mx.bfloat16
    print("Scoring alias and original source at block 32", flush=True)
    altered = evaluate(model, alias, cut=CUT)
    print("Scoring relocated definition at block 32", flush=True)
    definition = score_relocated(model, moved, cut=CUT)
    with np.load(root / PLACEMENT / "late.npz") as old:
        np.testing.assert_array_equal(definition["native"][..., :3], old["native"][..., :3])
    scores = dict(source=altered["early"], alias=altered["alias"], relocated=definition["whole"])
    eligible32 = good_groups(altered["native"]) & good_groups(definition["native"])
    local = metrics(scores, eligible32, 491627, ("source", "alias", "relocated"))
    with np.load(root / DEPTH / "depth.npz") as z:
        eligible72 = good_groups(z["native_alias"]) & good_groups(z["native_relocated"])
        same = eligible32 & eligible72
        cross = dict(at_32b=metrics(scores, same, 713511, ("source", "alias", "relocated")),
                     at_72b=metrics({"alias": z["alias_40"], "relocated": z["relocated_40"]},
                                    same, 713511, ("alias", "relocated")))
    n = local["shared"]
    control_open = (n >= 100 and local["switched"]["relocated"] >= .8*n
                    and local["constant_preserved"]["relocated"] >= .95*n)
    decision = "unresolved"
    if control_open and local["alias_minus_relocated"]["interval"][1] <= -.20:
        decision = "alias_gap_replicates_at_32b"
    elif control_open and local["switched"]["alias"] >= .50*n and local["constant_preserved"]["alias"] >= .95*n:
        decision = "32b_alias_switches_at_midpoint"
    np.savez_compressed(out / "midpoint.npz", **scores,
                        native_alias=altered["native"], native_relocated=definition["native"])
    (out / "programs.json").write_text(json.dumps(dict(alias=alias, relocated=moved), separators=(",", ":"))+"\n")
    report = dict(decision=decision, control_open=control_open, model=HOST_32, cut=CUT,
                  local=local, cross_host_shared=cross, source32_sha256=sha256(archived_rows).hexdigest(),
                  time_s=time.perf_counter()-begun, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
