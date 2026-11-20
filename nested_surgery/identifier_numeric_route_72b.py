"""Exchange a branch selector while leaving the host's two payloads intact."""
import argparse
from hashlib import sha256
from itertools import combinations
import gc
import json
import os
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, prefix
from identifier_binding import patch
from identifier_computation import DIGITS, options, programs
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST, edits
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

MODES = ("native", "donor", "orthogonal", "whole")
CONTEXTS = ("routing", "mirrored", "inert")


def recipes():
    # The target digits occur nowhere in the prompt as numeric literals.
    out = []
    for low, high in combinations(DIGITS, 2):
        for threshold in range(low + 1, high + 1):
            for inert in (low, high + 1):
                for offset in (-3, -2, -1, 1, 2, 3):
                    for first, second in options(offset):
                        literals = {low, high, threshold, inert, abs(offset), first, second}
                        if {first + offset, second + offset}.isdisjoint(literals):
                            out.append(((low, high), threshold, inert, offset, (first, second)))
    return out


def panel(tokenizer, count, start, *, lookup=False):
    rows = []
    selectors = []
    choices = recipes()
    if lookup:
        choices = sorted({(pair, pair[1], pair[0], offset, payloads)
                          for pair, _, _, offset, payloads in choices})
    for key in range(start, start + count):
        seed = f"numeric-lookup-72b-{key}" if lookup else f"numeric-router-72b-{key}"
        rng = random.Random(seed)
        pair, cut, inert, offset, payloads = rng.choice(choices)
        pair = rng.sample(pair, 2)
        first, second = rng.sample(payloads, 2)
        selectors.append(pair)
        selector = f"mode_{key}"
        left, right = f"left_{key}", f"right_{key}"
        operation = f" + {offset}" if offset > 0 else f" - {-offset}"
        body = "    if {} < {}:\n        return {}{}\n    return {}{}\n\nassert fetch() == "
        for context, (a, b, threshold) in zip(
            CONTEXTS, ((first, second, cut), (second, first, cut), (first, second, inert)), strict=True
        ):
            if lookup:
                low, high = sorted(pair)
                right_value = a if context == "inert" else b
                suffix = (f"    table = {{{low}: {a}, {high}: {right_value}}}\n"
                          f"    return table[{selector}]{operation}\n\nassert fetch() == ")
                answers = [str((a if v == low else right_value) + offset) for v in pair]
            else:
                rule = body.format(selector, threshold, left, operation, right, operation)
                suffix = f"    {left} = {a}\n    {right} = {b}\n{rule}"
                answers = [str((a if v < threshold else b) + offset) for v in pair]
            prompts = [f"def fetch():\n    {selector} = {v}\n{suffix}" for v in pair]
            assert (answers[0] == answers[1]) == (context == "inert")
            ids = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
            site, = [i for i, (u, v) in enumerate(zip(*ids, strict=True)) if u != v]
            assert [tokenizer.decode([seq[site]]).strip() for seq in ids] == list(map(str, pair))
            entire = [tokenizer.encode(p + a, add_special_tokens=False).ids
                      for p, a in zip(prompts, answers, strict=True)]
            assert [seq[:-1] for seq in entire] == ids
            for p, answer in zip(prompts, answers, strict=True):
                exec(compile(p + answer + "\n", "<selector probe>", "exec"), {})
            rows.append(dict(id=key, context=context, prompts=prompts, ids=ids, site=site,
                             answers=answers, choices=[seq[-1] for seq in entire],
                             labels=list(map(int, answers)), selector=pair, offset=offset,
                             threshold=None if lookup else threshold,
                             payloads=[a, right_value] if lookup else [a, b]))
    return rows, selectors


def good_groups(native):
    return (native[..., 0] == 1).reshape(-1, len(CONTEXTS), 2).all(axis=(1, 2))


def evaluate(model, rows, targets=None, geometry=None):
    records = {m: [] for m in (MODES if targets is not None else ("native",))}
    doses = []
    for i, (ids, site, ends, options, _) in enumerate(encoded_batches(rows)):
        h = prefix(model, ids, CUT)
        records["native"].extend(whole_vocabulary(model, h, ends, CUT, options))
        if targets is None:
            continue
        ix = mx.arange(len(ids))
        partner = ix ^ 1
        live = h[ix, site].astype(mx.float32)
        target = targets[i * 16:i * 16 + len(ids)][partner]
        narrow, sham = edits(live, target, *geometry)
        for mode, changed in (("donor", live + narrow), ("orthogonal", live + sham),
                              ("whole", live[partner])):
            records[mode].extend(whole_vocabulary(model, patch(h, site, changed), ends, CUT, options))
        doses.append(np.stack([np.array(mx.linalg.norm(bump, axis=1)) for bump in (narrow, sham)], axis=1))
        if (i + 1) % 12 == 0:
            print(f"Scored {(i + 1) * 8}/{len(rows)} contexts", flush=True)
    return {m: np.array(v) for m, v in records.items()}, np.concatenate(doses) if doses else None


def summarize(scores):
    eligible = good_groups(scores["native"])
    n = int(eligible.sum())
    measurements = {}
    joint = {}
    for m in MODES[1:]:
        ranks = scores[m].reshape(-1, len(CONTEXTS), 2, 4)[eligible]
        first, mirrored = (ranks[:, :2, :, 1] == 1).all(axis=2).T
        retained = (ranks[:, 2, :, 0] == 1).all(axis=1)
        joint[m] = first & mirrored & retained
        measurements[m] = dict(routing=int(first.sum()), mirrored=int(mirrored.sum()),
                               both_switch=int((first & mirrored).sum()), inert=int(retained.sum()),
                               joint=int(joint[m].sum()))
    contrast = None
    if n:
        draws = np.random.default_rng(597621).integers(n, size=(5000, n))
        contrast = interval(joint["donor"].astype(int) - joint["orthogonal"], draws)
    passed = (n >= 100 and min(measurements[m]["both_switch"] for m in ("donor", "whole")) >= .8 * n
              and measurements["donor"]["inert"] >= .95 * n
              and contrast["mean"] >= .2 and contrast["interval"][0] > 0)
    return dict(decision="selector_transfer" if passed else "unresolved", eligible=n, counts=measurements,
                donor_minus_orthogonal=contrast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", action="store_true")
    lookup = parser.parse_args().lookup
    starts = (168000, 169000, 170000) if lookup else (161000, 162000, 163000)
    began = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    prior = verified_result(jobs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    recipient = checkpoint_files(*HOST)
    tok = Tokenizer.from_file(str(recipient / "tokenizer.json"))
    screen, _ = panel(tok, 64, starts[0], lookup=lookup)
    model, _ = load_model(recipient, strict=True)
    assert len(model.model.layers) == 80 and model.args.hidden_size == 8192
    first, _ = evaluate(model, screen)
    eligible = int(good_groups(first["native"]).sum())
    print(f"Native routing qualification: {eligible}/64", flush=True)
    np.savez_compressed(out / "screen.npz", **first)
    (out / "screen.json").write_text(json.dumps(screen, separators=(",", ":")) + "\n")
    report = dict(decision="benchmark_below_floor", screen_eligible=eligible, screen_groups=64,
                  models={"donor": MODELS["donor"], "recipient": HOST}, cut=CUT, lookup=lookup)
    if eligible >= 48:
        del model
        gc.collect()
        rows, selectors = panel(tok, 256, starts[1], lookup=lookup)
        axes, digest = archived_maps(jobs / MAP_JOB)
        assert digest == prior["map_digest"]
        donor_axis, _, _, _, affine, bias = axes
        donor_path = model_files("donor")
        donor_tok = Tokenizer.from_file(str(donor_path / "tokenizer.json"))
        source = programs(donor_tok, len(selectors), starts[2], computed=False, values_by_pair=selectors)
        donor, _ = load_model(donor_path, strict=True)
        state = mx.array(gather(donor, source, SITES["donor"]).astype(np.float32))
        coordinates = (state @ donor_axis @ affine + bias).reshape(len(source), 2, -1)
        target = mx.repeat(coordinates, len(CONTEXTS), axis=0).reshape(len(rows) * 2, -1)
        mx.eval(target)
        del donor
        gc.collect()
        alignment = jobs / ALIGNMENT_JOB / "alignment.npz"
        keys = ("write", "read", "random", "center_input", "center_target")
        with np.load(alignment) as f:
            geometry = tuple(mx.array(f[k].astype(np.float32)) for k in keys)
        assert geometry[0].shape == (8192, 7)
        model, _ = load_model(recipient, strict=True)
        records, norms = evaluate(model, rows, target, geometry)
        assert all(ranks.shape == (len(rows), 2, 4) for ranks in records.values())
        np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
        np.savez_compressed(out / "selector.npz", norms=norms, **records)
        payload = json.dumps(dict(host=rows, donor=source), separators=(",", ":")) + "\n"
        (out / "programs.json").write_text(payload)
        report.update(groups=len(selectors), map_digest=digest, programs_digest=sha256(payload.encode()).hexdigest(),
                      alignment_digest=sha256(alignment.read_bytes()).hexdigest(), **summarize(records))
    report.update(runtime_s=time.perf_counter() - began, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
