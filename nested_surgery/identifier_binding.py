"""Use the same declaration edit with two different Python attribute reads."""

import json
import os
from pathlib import Path
import random
import string

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, finish, make_pairs, prefix
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

CLASS_NAMES = ("Amber", "Birch", "Cedar", "Delta", "Elm", "Flint", "Gold", "Harbor")
MODES = ("native", "selected", "unselected", "whole_site")


def programs(tokenizer, count=256, start=24000):
    rows = []
    for i in range(start, start + count):
        rng = random.Random(f"identifier-binding-v1-{i}")
        names = rng.sample(CLASS_NAMES, 2)
        letters = rng.sample(string.ascii_lowercase, 2)
        numbers = rng.sample(range(2, 97), 2)
        stem = f"item_{700000+i}"
        filler = "".join(f"    scratch_{j} = {rng.randrange(200, 300)}\n" for j in range(rng.randint(6, 12)))
        definitions = [
            "".join(f"class {name}:\n    {stem}{letter} = {number}\n\n"
                    for name, number in zip(names, numbers, strict=True))
            for letter in letters
        ]
        for query, (name, number) in enumerate(zip(names, numbers, strict=True)):
            prompts = [p + "def fetch():\n" + filler + f"    return {name}.{stem}" for p in definitions]
            ids = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
            complete = [tokenizer.encode(p+c, add_special_tokens=False).ids for p, c in zip(prompts, letters, strict=True)]
            assert [seq[:-1] for seq in complete] == ids
            sites = [j for j, (a, b) in enumerate(zip(*ids, strict=True)) if a != b]
            choices = [seq[-1] for seq in complete]
            assert len(sites) == 2 and choices[0] != choices[1]
            for version, p in enumerate(prompts):
                for candidate, letter in enumerate(letters):
                    env = {}
                    exec(compile(p+letter+"\n", "<generated binding probe>", "exec"), env)
                    try:
                        actual = env["fetch"]()
                    except AttributeError:
                        assert candidate != version
                    else:
                        assert candidate == version and actual == number
            rows.append(dict(id=i, query=query, ids=ids, prompts=prompts, letters=letters,
                             site=sites[0], sites=sites, choices=choices))
        first, second = rows[-2:]
        assert first["sites"] == second["sites"] and first["choices"] == second["choices"]
        for a, b in zip(first["ids"], second["ids"], strict=True):
            assert a[:first["sites"][-1]+1] == b[:second["sites"][-1]+1]
    return rows


def patch(h, positions, values):
    mask = (mx.arange(h.shape[1])[None, :] == positions[:, None])[..., None]
    return mx.where(mask, values[:, None, :].astype(h.dtype), h)


def evaluate(model, rows, axis, cut):
    axis = mx.array(axis)
    margins = {mode: [] for mode in MODES}
    doses = []
    offset = 0
    for ids, _, end, candidates, _ in encoded_batches(rows):
        batch = len(candidates)
        example = rows[offset:offset+batch//2]
        offset += len(example)
        sites = mx.array(np.repeat([row["sites"] for row in example], 2, axis=0))
        query = mx.array(np.repeat([row["query"] for row in example], 2))
        index = mx.arange(batch)
        canonical = mx.array(np.arange(batch) & ~2)
        partner = mx.array(np.arange(batch) ^ 1)
        h = prefix(model, ids, cut)
        state = h[index[:, None], sites].astype(mx.float32)
        np.testing.assert_allclose(np.array(state), np.array(state[canonical]), atol=0, rtol=0)
        original = state[canonical]
        delta = original[partner] - original
        projected = (delta @ axis) @ axis.T
        norm = np.array(mx.linalg.norm(projected, axis=-1)).reshape(-1, 2, 2, 2)
        np.testing.assert_allclose(norm[:, 0], norm[:, 1], atol=0, rtol=0)
        doses.extend(norm)
        margins["native"].extend(finish(model, h, end, cut, candidates))
        for name, which in (("selected", query), ("unselected", 1-query)):
            changed = state[index, which] + projected[index, which]
            margins[name].extend(finish(model, patch(h, sites[index, which], changed), end, cut, candidates))
        full = original[partner][index, query]
        margins["whole_site"].extend(finish(model, patch(h, sites[index, query], full), end, cut, candidates))
    scores = {k: np.asarray(v).reshape(-1, 2, 2) for k, v in margins.items()}
    assert all(np.isfinite(v).all() for v in scores.values())
    return scores, np.asarray(doses)


def summarize(scores):
    eligible = (scores["native"] > 0).all(axis=(1, 2))
    n = int(eligible.sum())
    flips = {k: (v[eligible] < 0).all(axis=2) for k, v in scores.items() if k != "native"}
    counts = {k: v.sum(axis=0).tolist() for k, v in flips.items()}
    means = {k: float(v.mean()) if n else None for k, v in flips.items()}
    if n:
        draws = np.random.default_rng(875024).integers(n, size=(5000,n))
        contrast = interval(flips["selected"].mean(1)-flips["unselected"].mean(1), draws)
    else:
        contrast = None
    supported = (n >= 100 and means["whole_site"] >= .5 and means["selected"] >= .5
                 and means["unselected"] <= .1 and contrast["interval"][0] > .25)
    return dict(decision="binding_sensitive" if supported else "unresolved",
                eligible_groups=n, reversals_by_site=counts, reversal_rates=means, selected_minus_unselected=contrast)


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    path = model_files("host")
    tokenizer = Tokenizer.from_file(str(path/"tokenizer.json"))
    calibration = make_pairs([tokenizer], 512, 0)
    rows = programs(tokenizer)
    model, _ = load_model(path, strict=True)
    cut = SITES["host"]
    labels = [ord(c)-97 for row in calibration for c in row["letters"]]
    axis, _ = bases(gather(model, calibration, cut), labels)
    scores, doses = evaluate(model, rows, axis, cut)
    report = summarize(scores)
    np.savez_compressed(out/"binding.npz", perturbation_norms=doses, axis=axis, **scores)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(dict(model=MODELS["host"],cut=cut,groups=len(rows)//2,
                                                  calibration_pairs=len(calibration),rank=25,**report),indent=2)+"\n")


if __name__ == "__main__":
    main()
