"""Apply the already fitted 72B alignment simultaneously to two operand sites."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_numeric_carry import finish_tokens, host_panel, outcomes, predict, summarize
from identifier_numeric_carry_32b import (
    CARRY_JOB, CARRY_REVISION, gather_targets, verified_result,
)
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_numeric_width_72b import HOST
from large_opener import MODELS, checkpoint_files, model_files

ALIGNMENT_JOB = "qwen-cross-width-72b"
ALIGNMENT_REVISION = "d2609dcba899b8755b47a8862512aca8e8401326"
CUT = 17


def edits(state, target, write, read, random, input_mean, coordinate_mean):
    difference = target - ((state - input_mean) @ read + coordinate_mean)
    altered = difference @ write.T
    sham = difference @ random.T
    norm = mx.linalg.norm(altered, axis=-1, keepdims=True)
    base_norm = mx.linalg.norm(sham, axis=-1, keepdims=True)
    return altered, sham * (norm / mx.maximum(base_norm, 1e-12))


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    verified_result(jobs / CARRY_JOB, CARRY_REVISION, "carry_composes")
    alignment_result = verified_result(jobs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    original = jobs / CARRY_JOB
    payload = (original / "programs.json").read_bytes()
    source = json.loads(payload)
    panel = source["host"]
    mapping, digest = archived_maps(jobs / MAP_JOB)
    assert digest == alignment_result["map_digest"]
    donor_basis, _, _, _, affine, bias = mapping
    desired = gather_targets(model_files("donor"), source, panel, donor_basis, affine, bias)
    path = checkpoint_files(*HOST)
    token = Tokenizer.from_file(str(path / "tokenizer.json"))
    screen = json.loads((original / "screen.json").read_text())
    assert screen == host_panel(token, 128, 110000)
    assert panel == host_panel(token, 512, 111000)
    model, _ = load_model(path, strict=True)
    assert len(model.model.layers) == 80 and model.args.hidden_size == 8192
    first = predict(model, screen, cut=CUT)
    endings = finish_tokens(token, np.unique(first["native"][..., 0]))
    screen_count = int(outcomes(screen, first, endings)["native"].all(1).sum())
    np.savez_compressed(out / "screen.npz", terminators=np.array(sorted(endings)), **first)
    report = dict(models={"donor": MODELS["donor"], "receiver": HOST}, screen_eligible=screen_count,
                  screen_groups=len(screen), decision="benchmark_below_floor", cut=CUT)
    print(f"Native 72B carry qualification {screen_count}/{len(screen)}", flush=True)
    if screen_count >= 64:
        prior = jobs / ALIGNMENT_JOB / "alignment.npz"
        with np.load(prior) as z:
            keys = ("write", "read", "random", "center_input", "center_target")
            geometry = tuple(mx.array(z[k].astype(np.float32)) for k in keys)
        assert geometry[0].shape == (8192, 7)
        norms = {"both": [], "random": []}

        def edit(state, start, count, mode):
            target = desired[start:start + count].reshape(len(state), 2, -1)
            changed = edits(state, target, *geometry)[int(mode == "random")]
            norms[mode].append(np.array(mx.linalg.norm(changed, axis=-1)))
            return (state + changed).transpose(1, 0, 2)

        scores = predict(model, panel, cut=CUT, operand_edit=edit)
        assert all(data.shape == (512, 4, 3, 2) for data in scores.values())
        doses = {m: np.concatenate(rows) for m, rows in norms.items()}
        np.testing.assert_allclose(doses["both"], doses["random"], rtol=1e-5, atol=1e-6)
        endings = finish_tokens(token, np.unique(np.concatenate([v[..., 0].ravel() for v in scores.values()])))
        report.update(groups=len(panel), map_digest=digest, prompts_digest=sha256(payload).hexdigest(),
                      alignment_digest=sha256(prior.read_bytes()).hexdigest(),
                      **summarize(panel, outcomes(panel, scores, endings)))
        np.savez_compressed(out / "carry-72b.npz", **scores, both_norm=doses["both"], random_norm=doses["random"],
                            terminators=np.array(sorted(endings)))
    report.update(runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
