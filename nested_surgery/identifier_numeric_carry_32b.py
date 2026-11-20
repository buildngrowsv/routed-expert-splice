"""Test whether an unchanged borrowed 32B gain composes across two operands."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES
from identifier_numeric_carry import finish_tokens, host_panel, outcomes, predict, summarize
from identifier_numeric_scale import HOST_32, PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

CARRY_JOB = "qwen-two-operands-carry"
CARRY_REVISION = "c4062f2af3d71f0167688c21b7bd786c9bc244b3"
GAIN_JOB = "qwen-one-pass-scalar-32b"
GAIN_REVISION = "f00e012434a769932da68c89e832b43644f1b1f4"
CUT = 13


def verified_result(job, revision, decision):
    receipt = json.loads((job / "receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["revision"] == revision
    result = json.loads((job / "result.json").read_text())
    assert result["decision"] == decision
    return result


def edit_both(state, targets, host_axis, random_axis, gain):
    delta = targets - state @ host_axis
    transplanted = gain * (delta @ host_axis.T)
    orthogonal = gain * (delta @ random_axis.T)
    wanted = mx.linalg.norm(transplanted, axis=-1, keepdims=True)
    actual = mx.linalg.norm(orthogonal, axis=-1, keepdims=True)
    assert bool(mx.all((actual > 0) | (wanted == 0)).item())
    orthogonal = orthogonal * (wanted / mx.maximum(actual, 1e-12))
    return transplanted, orthogonal


def gather_targets(donor_path, source, panel, donor_axis, affine, bias):
    donor, _ = load_model(donor_path, strict=True)
    targets = []
    for name, pattern in (("left", (1, 1, 0, 0)), ("right", (1, 0, 1, 0))):
        activation = mx.array(gather(donor, source["donor"][name], SITES["donor"]).astype(np.float32))
        target = (activation @ donor_axis @ affine + bias).reshape(len(panel), 2, -1)
        targets.append(mx.take(target, mx.array(pattern), axis=1))
    desired = mx.stack(targets, axis=2)
    mx.eval(desired)
    del donor
    gc.collect()
    return desired


def main():
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    donor_path = model_files("donor")
    path = checkpoint_files(*HOST_32)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    old = root / CARRY_JOB
    verified_result(old, CARRY_REVISION, "carry_composes")
    scaling = verified_result(root / GAIN_JOB, GAIN_REVISION, "answer_free_gain")
    gain = scaling["gain"]
    assert np.isfinite(gain) and gain > 0
    model, _ = load_model(path, strict=True)
    assert len(model.model.layers) == 64
    screen = json.loads((old / "screen.json").read_text())
    assert screen == host_panel(tok, 128, 110000)
    first = predict(model, screen, cut=CUT)
    endings = finish_tokens(tok, np.unique(first["native"][..., 0]))
    eligible = int(outcomes(screen, first, endings)["native"].all(1).sum())
    np.savez_compressed(out / "screen.npz", terminators=np.array(sorted(endings)), **first)
    print(f"Native 32B carry qualification {eligible}/128", flush=True)
    report = dict(decision="benchmark_below_floor", screen_eligible=eligible, screen_groups=128,
                  models={"donor": MODELS["donor"], "32b": HOST_32}, gain=gain)
    if eligible >= 64:
        del model
        gc.collect()
        payload = (old / "programs.json").read_bytes()
        source = json.loads(payload)
        panel = source["host"]
        assert panel == host_panel(tok, 512, 111000)
        mapping, digest = archived_maps(root / MAP_JOB)
        donor_axis, host_axis, random_axis, _, affine, bias = mapping
        assert digest == scaling["map_digest"]
        desired = gather_targets(donor_path, source, panel, donor_axis, affine, bias)

        norms = {"both": [], "random": []}
        def edit(state, start, count, mode):
            target = desired[start:start + count].reshape(len(state), 2, -1)
            index = int(mode == "random")
            changes = edit_both(state, target, host_axis, random_axis, gain)[index]
            norms[mode].append(np.array(mx.linalg.norm(changes, axis=-1)))
            return (state + changes).transpose(1, 0, 2)

        model, _ = load_model(path, strict=True)
        scores = predict(model, panel, cut=CUT, operand_edit=edit)
        assert all(a.shape == (512, 4, 3, 2) for a in scores.values())
        norm_data = {m: np.concatenate(x) for m, x in norms.items()}
        np.testing.assert_allclose(norm_data["both"], norm_data["random"], rtol=1e-5, atol=1e-6)
        endings = finish_tokens(tok, np.unique(np.concatenate([x[..., 0].ravel() for x in scores.values()])))
        report.update(groups=len(panel), map_digest=digest, prompts_digest=sha256(payload).hexdigest(),
                      **summarize(panel, outcomes(panel, scores, endings)))
        np.savez_compressed(out / "carry-32b.npz", **scores, both_norm=norm_data["both"],
                            random_norm=norm_data["random"], terminators=np.array(sorted(endings)))
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
