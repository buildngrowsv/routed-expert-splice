"""Confirm the late-key donor effect on disjoint paired lookup programs."""
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from identifier_baseline import SITES
from identifier_computation import programs
from identifier_lookup_order_72b import move_key
from identifier_lookup_order_transfer_72b import summarize
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST
from identifier_numeric_route_72b import evaluate, panel
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

CONFIRM = "qwen-late-key-donor-transfer-72b"
CONFIRM_REVISION = "436d2c7edfa8b3ab6f103cd22678f61d1d5633b7"


def main():
    begun = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    prior = verified_result(jobs / CONFIRM, CONFIRM_REVISION, "late_key_donor_transfer")
    bridge = verified_result(jobs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    host_path, donor_path = checkpoint_files(*HOST), model_files("donor")
    host_tok = Tokenizer.from_file(str(host_path / "tokenizer.json"))
    donor_tok = Tokenizer.from_file(str(donor_path / "tokenizer.json"))
    early, values = panel(host_tok, 256, 176000, lookup=True)
    late = move_key(host_tok, early)
    inputs = programs(donor_tok, len(values), 177000, computed=False, values_by_pair=values)
    mapping, map_hash = archived_maps(jobs / MAP_JOB)
    assert map_hash == prior["map_digest"] == bridge["map_digest"]
    donor_axis, _, _, _, affine, intercept = mapping
    donor, _ = load_model(donor_path, strict=True)
    states = mx.array(gather(donor, inputs, SITES["donor"]).astype(np.float32))
    vectors = (states @ donor_axis @ affine + intercept).reshape(len(inputs), 2, -1)
    targets = mx.repeat(vectors, 3, axis=0).reshape(len(early)*2, -1)
    mx.eval(targets)
    del donor
    gc.collect()
    alignment = jobs / ALIGNMENT_JOB / "alignment.npz"
    alignment_hash = sha256(alignment.read_bytes()).hexdigest()
    assert alignment_hash == prior["alignment_digest"]
    with np.load(alignment) as f:
        geometry = tuple(mx.array(f[k].astype(np.float32))
                         for k in ("write", "read", "random", "center_input", "center_target"))
    model, _ = load_model(host_path, strict=True)
    endpoints = {}
    for name, rows in (("early", early), ("late", late)):
        print(f"Starting {name}-key contexts", flush=True)
        scores, norms = evaluate(model, rows, targets, geometry)
        np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
        np.savez_compressed(out / f"{name}.npz", norms=norms, **scores)
        endpoints[name] = scores
    archive = json.dumps(dict(early=early, late=late, donor=inputs), separators=(",", ":")) + "\n"
    (out / "programs.json").write_text(archive)
    result = summarize(endpoints["early"], endpoints["late"])
    result.update(models={"donor":MODELS["donor"],"recipient":HOST}, cut=CUT, map_digest=map_hash,
                  alignment_digest=alignment_hash, programs_digest=sha256(archive.encode()).hexdigest(),
                  runtime_s=time.perf_counter()-begun, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
