"""Test whether the fitted donor follows the recovered late dictionary key."""
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES
from identifier_computation import programs
from identifier_lookup_order_72b import EARLY_JOB, EARLY_REVISION, move_key
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST
from identifier_numeric_route_72b import evaluate, good_groups
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

LATE_JOB = "qwen-dictionary-key-order-72b"
LATE_REVISION = "2ceb1b12fdc7238065394c273ebe6be2936ed1d8"


def summarize(early, late):
    eligible = good_groups(early["native"]) & good_groups(late["native"])
    n = int(eligible.sum())
    switches, inert = {}, {}
    data = {"early_donor": early["donor"], "late_donor": late["donor"],
            "late_orthogonal": late["orthogonal"], "late_whole": late["whole"]}
    for mode, ranks in data.items():
        z = ranks.reshape(-1, 3, 2, 4)[eligible]
        switches[mode] = (z[:, :2, :, 1] == 1).all(axis=(1, 2))
        inert[mode] = int((z[:, 2, :, 0] == 1).all(axis=1).sum())
    counts = {m: int(v.sum()) for m, v in switches.items()}
    effects = {}
    if n:
        draws = np.random.default_rng(876211).integers(n, size=(5000, n))
        for comparison in ("early_donor", "late_orthogonal", "late_whole"):
            effects[comparison] = interval(switches["late_donor"].astype(int) - switches[comparison], draws)
    passed = (n >= 100 and min(counts[m] for m in ("late_donor", "late_whole")) >= .8*n
              and inert["late_donor"] >= .95*n and effects["early_donor"]["mean"] >= .15
              and effects["early_donor"]["interval"][0] > 0 and effects["late_orthogonal"]["mean"] >= .2
              and effects["late_orthogonal"]["interval"][0] > 0)
    return dict(decision="late_key_donor_transfer" if passed else "unresolved",
                eligible=n, mirrored_switches=counts, constant_preserved=inert, late_donor_minus=effects)


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    first = verified_result(jobs / EARLY_JOB, EARLY_REVISION, "unresolved")
    latest = verified_result(jobs / LATE_JOB, LATE_REVISION, "key_order_restores_whole_state_routing")
    bridge = verified_result(jobs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    raw = (jobs / EARLY_JOB / "programs.json").read_bytes()
    digest = sha256(raw).hexdigest()
    assert digest == first["programs_digest"] == latest["original_programs_digest"]
    source = json.loads(raw)
    host_path = checkpoint_files(*HOST)
    tokenizer = Tokenizer.from_file(str(host_path / "tokenizer.json"))
    rows = move_key(tokenizer, source["host"])
    assert rows == json.loads((jobs / LATE_JOB / "programs.json").read_text())
    mapping, map_hash = archived_maps(jobs / MAP_JOB)
    assert map_hash == first["map_digest"] == bridge["map_digest"]
    donor_axis, _, _, _, affine, intercept = mapping
    donor, _ = load_model(model_files("donor"), strict=True)
    states = mx.array(gather(donor, source["donor"], SITES["donor"]).astype(np.float32))
    vectors = (states @ donor_axis @ affine + intercept).reshape(len(rows)//3, 2, -1)
    target = mx.repeat(vectors, 3, axis=0).reshape(len(rows)*2, -1)
    mx.eval(target)
    del donor
    gc.collect()
    alignment = jobs / ALIGNMENT_JOB / "alignment.npz"
    alignment_hash = sha256(alignment.read_bytes()).hexdigest()
    assert alignment_hash == first["alignment_digest"]
    with np.load(alignment) as f:
        geometry = tuple(mx.array(f[k].astype(np.float32))
                         for k in ("write", "read", "random", "center_input", "center_target"))
    model, _ = load_model(host_path, strict=True)
    late, doses = evaluate(model, rows, target, geometry)
    np.testing.assert_allclose(doses[:, 0], doses[:, 1], rtol=1e-5, atol=1e-6)
    with np.load(jobs / LATE_JOB / "moved-key.npz") as prior:
        for m in ("native", "whole"):
            np.testing.assert_array_equal(late[m][..., :3], prior[m][..., :3])
    prior_ranks = jobs / EARLY_JOB / "selector.npz"
    assert sha256(prior_ranks.read_bytes()).hexdigest() == latest["original_ranks_digest"]
    with np.load(prior_ranks) as f:
        early = {k: f[k] for k in ("native", "donor")}
    report = summarize(early, late)
    assert report["eligible"] == latest["shared_native_groups"]
    np.savez_compressed(out / "late-donor.npz", norms=doses, **late)
    report.update(models={"donor": MODELS["donor"], "recipient": HOST}, cut=CUT, map_digest=map_hash,
                  alignment_digest=alignment_hash, original_programs_digest=digest,
                  runtime_s=time.perf_counter()-start, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
