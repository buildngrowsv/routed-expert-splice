"""Carry both frozen spliced operand projections through seven additional blocks."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_numeric_carry import finish_tokens, outcomes, predict
from identifier_numeric_carry_32b import (
    CARRY_JOB, CARRY_REVISION, CUT, GAIN_JOB, GAIN_REVISION,
    edit_both, gather_targets, verified_result,
)
from identifier_numeric_composition import replace
from identifier_numeric_scale import HOST_32, PRIOR as MAP_JOB, archived_maps
from large_opener import checkpoint_files, model_files

HELD = ("native", "both", "random")
PRIOR = "qwen-borrowed-two-operands-32b"
REVISION = "f0fc062c821766701472077421ec9cc02c6d11e6"
STEPS = 7
ABSOLUTE_JOB = "qwen-fixed-two-operands-32b"
ABSOLUTE_REVISION = "708db914683b86fcb7b1acf02fca18fdfd87131f"


def restore_projection(hidden, sites, axis, target):
    rows = mx.arange(len(hidden))[:, None]
    state = hidden[rows, sites].astype(mx.float32)
    correction = (target - state @ axis) @ axis.T
    return replace(hidden, sites, (state + correction).transpose(1, 0, 2))


def fixed_projection(initial, sites, axis):
    index = mx.arange(len(initial))[:, None]
    reference = initial[index, sites].astype(mx.float32) @ axis
    mx.eval(reference)

    def restore(hidden):
        return restore_projection(hidden, sites, axis, reference)

    return STEPS, restore


def relative_projection(model, initial, original, sites, axis):
    rows = mx.arange(len(initial))[:, None]

    def coordinates(hidden):
        return hidden[rows, sites].astype(mx.float32) @ axis

    offset = coordinates(initial) - coordinates(original)
    mask = create_attention_mask(original, None)
    reference = []
    for layer in model.model.layers[CUT:CUT + STEPS]:
        original = layer(original, mask)
        reference.append(coordinates(original) + offset)
    mx.eval(*reference)
    targets = iter(reference)
    return STEPS, lambda hidden: restore_projection(hidden, sites, axis, next(targets))


def relative_contrast(held, one_shot, absolute):
    eligible = held["native"].all(1) & one_shot["native"].all(1) & absolute["native"].all(1)
    n = int(eligible.sum())
    wins = {name: arm[eligible].all(1) for name, arm in (
        ("relative", held["both"]), ("relative_random", held["random"]),
        ("absolute", absolute["both"]), ("one_shot", one_shot["both"]))}
    counts = {name: int(hits.sum()) for name, hits in wins.items()}
    differences = {}
    if n:
        draws = np.random.default_rng(628177).integers(n, size=(5000, n))
        differences = {name: interval(wins["relative"].astype(int) - hits, draws)
                       for name, hits in wins.items() if name != "relative"}
    enough = n >= .95 * one_shot["native"].all(1).sum()
    rescued = bool(n and enough and differences["absolute"]["mean"] >= .04
                   and differences["absolute"]["interval"][0] > 0
                   and differences["one_shot"]["interval"][0] >= -.04
                   and differences["relative_random"]["mean"] >= .20
                   and differences["relative_random"]["interval"][0] > 0)
    return dict(decision="relative_hold_recovers" if rescued else "unresolved",
                common_native=n, common_counts=counts, relative_minus=differences)


def summarize(held, one_shot, panel):
    base = one_shot["native"].all(1)
    healthy = held["native"].all(1)
    selected = base & healthy
    n = int(selected.sum())
    success = {"hold": held["both"], "sham": held["random"],
               "one_shot": one_shot["both"], "whole": one_shot["whole"]}
    result = {name: flag[selected].all(1) for name, flag in success.items()}
    counts = {name: int(flag.sum()) for name, flag in result.items()}
    contrasts = {}
    if n:
        draws = np.random.default_rng(781394).integers(n, size=(5000, n))
        contrasts = {name: interval(result["hold"].astype(int) - result[name], draws)
                     for name in ("one_shot", "sham")}
    lengths = np.array([[len(x) for x in row["answers"]] for row in panel])
    directions = {}
    for name, a, b in (("one_to_two", 1, 2), ("two_to_one", 2, 1)):
        included = selected[:, None] & (lengths == a) & (lengths[:, ::-1] == b)
        directions[name] = dict(total=int(included.sum()),
                                correct={arm: int(v[included].sum()) for arm, v in success.items()})
    good = bool(n >= .95 * base.sum() and contrasts and contrasts["one_shot"]["mean"] >= .05
                and contrasts["one_shot"]["interval"][0] > 0 and contrasts["sham"]["mean"] >= .20
                and contrasts["sham"]["interval"][0] > 0)
    return dict(decision="held_reference_improves" if good else "unresolved",
                native=int(base.sum()), native_held=int(healthy.sum()), eligible=n, counts=counts,
                contrasts=contrasts, directions=directions, hit_original_target=bool(n and counts["hold"] >= .8*n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=("absolute", "relative"), default="absolute")
    args = parser.parse_args()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    previous = verified_result(jobs / PRIOR, REVISION, "unresolved")
    assert previous["eligible"] >= 100
    initial = verified_result(jobs / GAIN_JOB, GAIN_REVISION, "answer_free_gain")
    verified_result(jobs / CARRY_JOB, CARRY_REVISION, "carry_composes")
    payload = (jobs / CARRY_JOB / "programs.json").read_bytes()
    digest = sha256(payload).hexdigest()
    assert digest == previous["prompts_digest"]
    source = json.loads(payload)
    panel = source["host"]
    mapping, map_digest = archived_maps(jobs / MAP_JOB)
    assert map_digest == previous["map_digest"] == initial["map_digest"]
    donor_axis, host_axis, orthogonal, _, affine, bias = mapping
    gain = initial["gain"]
    targets = gather_targets(model_files("donor"), source, panel, donor_axis, affine, bias)
    path = checkpoint_files(*HOST_32)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    model, _ = load_model(path, strict=True)

    def edit(state, start, count, mode):
        desired = targets[start:start + count].reshape(len(state), 2, -1)
        changes = edit_both(state, desired, host_axis, orthogonal, gain)[int(mode == "random")]
        return (state + changes).transpose(1, 0, 2)

    def hold(initial_state, original_state, sites, mode):
        axis = orthogonal if mode == "random" else host_axis
        if args.reference == "relative":
            return relative_projection(model, initial_state, original_state, sites, axis)
        return fixed_projection(initial_state, sites, axis)

    scores = predict(model, panel, cut=CUT, operand_edit=edit, hold_factory=hold, modes=HELD)
    endings = finish_tokens(tok, np.unique(np.concatenate([v[..., 0].ravel() for v in scores.values()])))
    with np.load(jobs / PRIOR / "carry-32b.npz") as prior:
        old = outcomes(panel, {m: prior[m] for m in ("native", "both", "random", "whole")}, prior["terminators"])
    current = outcomes(panel, scores, endings)
    result = summarize(current, old, panel)
    if args.reference == "relative":
        held = verified_result(jobs / ABSOLUTE_JOB, ABSOLUTE_REVISION, "unresolved")
        assert held["prompts_digest"] == digest and held["map_digest"] == map_digest
        with np.load(jobs / ABSOLUTE_JOB / "held-32b.npz") as archive:
            absolute = outcomes(panel, {m: archive[m] for m in HELD}, archive["terminators"])
        result.update(relative_contrast(current, old, absolute))
    np.savez_compressed(out / "held-32b.npz", **scores, terminators=np.array(sorted(endings)))
    report = dict(host=HOST_32, reference=args.reference, gain=gain, hold_blocks=STEPS, prompts_digest=digest, map_digest=map_digest,
                  peak_mlx_bytes=mx.get_peak_memory(), **result)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
