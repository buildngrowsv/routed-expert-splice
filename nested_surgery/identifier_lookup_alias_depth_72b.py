"""Test alias-state sufficiency at deeper cuts with a relocated-definition control."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np

from identifier_baseline import encoded_batches
from identifier_binding import patch
from identifier_lookup_alias_72b import FRESH, REVISION
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import CUT, HOST
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files

ALIAS = "qwen-dictionary-late-alias-site-72b"
ALIAS_REVISION = "a2e74ee1055c172ae6a1e02903da7610df6fb37a"
DEEPER = (40, 60)


def evaluate(model, rows, sites, old):
    results = {}
    for batch, (ids, source, ends, choices, _) in enumerate(encoded_batches(rows)):
        h = model.model.embed_tokens(ids)
        mask = create_attention_mask(h, None)
        partner = mx.arange(len(ids)) ^ 1
        row_slice = slice(batch * 8, batch * 8 + len(ids) // 2)
        locations = {"source": source}
        if "alias" in sites:
            locations["alias"] = mx.array(np.repeat([r["alias_site"] for r in rows[row_slice]], 2))
        # Replay the first batch at the original cut before scoring the new cuts.
        boundaries = DEEPER if batch else (CUT, *DEEPER)
        previous = 0
        for cut in boundaries:
            for block in model.model.layers[previous:cut]:
                h = block(h, mask)
            mx.eval(h)
            previous = cut
            for name, where in locations.items():
                branch = patch(h, where, h[partner, where])
                ranks = whole_vocabulary(model, branch, ends, cut, choices)
                if cut == CUT:
                    np.testing.assert_array_equal(ranks[..., :3], old[name][row_slice, :, :3])
                else:
                    results.setdefault(f"{name}_{cut}", []).append(ranks)
            if cut == CUT:
                untouched = whole_vocabulary(model, h, ends, cut, choices)
                np.testing.assert_array_equal(untouched[..., :3], old["native"][row_slice, :, :3])
        if (batch + 1) % 12 == 0:
            print(f"Completed {len(sites)}-site depth test on {(batch + 1) * 8}/{len(rows)} contexts", flush=True)
    return {k: np.concatenate(v) for k, v in results.items()}


def effect(a, b, indices):
    delta = a.astype(int) - b
    interval = np.percentile(delta[indices].mean(axis=1), [1.25, 98.75])
    return dict(mean=float(delta.mean()), interval=interval.tolist())


def summarize(native, scores):
    shared = good_groups(native["alias"]) & good_groups(native["relocated"])
    n = int(shared.sum())
    metrics, switches = {}, {}
    for key, ranks in scores.items():
        panels = ranks.reshape(-1, 3, 2, 4)[shared]
        switches[key] = (panels[:, :2, :, 1] == 1).all(axis=(1, 2))
        metrics[key] = dict(switch=int(switches[key].sum()),
                            preserve=int((panels[:, 2, :, 0] == 1).all(axis=1).sum()))
    if n < 100:
        return dict(decision="insufficient_shared_coverage", shared=n, metrics=metrics)
    samples = np.random.default_rng(176614).integers(n, size=(8000, n))
    contrasts, viable, recovered = {}, [], []
    for cut in DEEPER:
        alias, relocation = f"alias_{cut}", f"relocated_{cut}"
        change = effect(switches[alias], switches[f"alias_{CUT}"], samples)
        gap = effect(switches[alias], switches[relocation], samples)
        contrasts[str(cut)] = {"alias_minus_initial": change, "alias_minus_relocated": gap}
        if metrics[relocation]["switch"] >= .8*n and metrics[relocation]["preserve"] >= .95*n:
            viable.append(cut)
            if (metrics[alias]["switch"] >= .5*n and metrics[alias]["preserve"] >= .95*n
                    and change["mean"] >= .2 and change["interval"][0] > 0):
                recovered.append(cut)
    decision = "unresolved"
    if recovered:
        decision = "alias_recovers_with_depth"
    elif viable and all(contrasts[str(c)]["alias_minus_relocated"]["interval"][1] < -.2 for c in viable):
        decision = "alias_gap_persists_at_viable_cuts"
    return dict(decision=decision, shared=n, metrics=metrics, contrasts=contrasts,
                viable_cuts=viable, recovered_cuts=recovered)


def main():
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    first, copy = runs / FRESH, runs / ALIAS
    placed = verified_result(first, REVISION, "late_key_donor_transfer")
    duplicate = verified_result(copy, ALIAS_REVISION, "unresolved")
    original = (first / "programs.json").read_bytes()
    assert sha256(original).hexdigest() == placed["programs_digest"] == duplicate["source_programs_digest"]
    alias_payload = (copy / "programs.json").read_bytes()
    rows = {"alias": json.loads(alias_payload), "relocated": json.loads(original)["late"]}
    assert [r["id"] for r in rows["alias"]] == [r["id"] for r in rows["relocated"]]
    assert [r["answers"] for r in rows["alias"]] == [r["answers"] for r in rows["relocated"]]
    inputs = {"alias": copy / "alias.npz", "relocated": first / "late.npz"}
    baselines = {}
    for name, p in inputs.items():
        with np.load(p) as f:
            baselines[name] = {k: f[k] for k in ("native", "early", "alias")} if name == "alias" else {
                "native": f["native"], "source": f["whole"]}
    baselines["alias"]["source"] = baselines["alias"].pop("early")
    model, _ = load_model(checkpoint_files(*HOST), strict=True)
    assert len(model.model.layers) == 80 and model.model.embed_tokens.weight.dtype == mx.bfloat16
    outputs = {}
    for form, sites in (("alias", ("source", "alias")), ("relocated", ("source",))):
        print(f"Scoring the {form} layout", flush=True)
        initial = {k: baselines[form][k] for k in sites}
        deeper = evaluate(model, rows[form], sites, baselines[form])
        combined = {f"{k}_{CUT}": v for k, v in initial.items()} | deeper
        for mode, v in combined.items():
            key = f"{form}_{mode.split('_')[-1]}" if form == "relocated" else mode
            outputs[key] = v
    native = {k: v["native"] for k, v in baselines.items()}
    np.savez_compressed(out / "depth.npz", **outputs, **{f"native_{k}": v for k, v in native.items()})
    report = summarize(native, outputs)
    report.update(model=HOST, cuts=(CUT, *DEEPER), raw_inputs={k: sha256(p.read_bytes()).hexdigest()
                  for k, p in inputs.items()}, alias_programs_sha256=sha256(alias_payload).hexdigest(),
                  original_programs_sha256=sha256(original).hexdigest(), runtime_s=time.perf_counter()-started,
                  peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
