"""Test whether the same proposed operands obey two opposite host expressions."""
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_computation import programs
from identifier_numeric_carry import finish_tokens, host_panel, outcomes, predict
from identifier_numeric_carry_32b import gather_targets, verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST, edits
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from large_opener import MODELS, checkpoint_files, model_files

PRIOR = "qwen-aligned-two-operands-72b"
REVISION = "e240a141e91e198a3729870d54822cdbef7a0403"


def crossed_panel(tok, count, start):
    original = host_panel(tok, count, start)
    crossed = []
    for row in original:
        x, y = f"left_{800000+row['id']}", f"right_{800000+row['id']}"
        old = f"    return {x} + {y}\n"
        values = list(product(row["left"], row["right"]))
        for sign, formula in ((1, f"20 + {x} - {y}"), (-1, f"20 - {x} + {y}")):
            prompts = [p.replace(old, f"    return {formula}\n") for p in row["prompts"]]
            assert all(a != b for a, b in zip(prompts, row["prompts"], strict=True))
            ans = [str(20 + sign*(u-v)) for u,v in values]
            assert len(set(ans)) == 4 and all(len(v) == 2 for v in ans)
            ids = [tok.encode(p, add_special_tokens=False).ids for p in prompts]
            sites = [i for i, pieces in enumerate(zip(*ids, strict=True)) if len(set(pieces)) != 1]
            encodings = [tok.encode(v, add_special_tokens=False).ids for v in ans]
            assert sites == row["sites"]
            assert all(len(e) == 2 for e in encodings)
            for prompt, prefix, output, encoded in zip(prompts, ids, ans, encodings, strict=True):
                exec(compile(prompt + output + "\n", "<generated operator assertion>", "exec"), {})
                assert tok.encode(prompt + output, add_special_tokens=False).ids == prefix + encoded
            crossed.append(dict(row, prompts=prompts, ids=ids, sites=sites, answers=ans, answer_tokens=encodings,
                                sign=sign))
    return original, crossed


def count(hits):
    return {m: x.reshape(-1, 2, 4).all(axis=2) for m, x in hits.items()}


def summarize(hits):
    per_context = count(hits)
    qualifies = per_context["native"].all(axis=1)
    n = int(qualifies.sum())
    paired = {m: x[qualifies].all(axis=1) for m, x in per_context.items()}
    totals = {m: int(x.sum()) for m, x in paired.items()}
    score = None
    if n:
        draws = np.random.default_rng(431689).integers(n, size=(5000, n))
        score = interval(paired["both"].astype(int) - paired["random"], draws)
    passed = bool(n >= 100 and min(totals[k] for k in ("both", "whole")) >= .8*n
                  and score["mean"] >= .2 and score["interval"][0] > 0)
    return dict(decision="donor_obeys_host_operator" if passed else "unresolved", eligible=n, complete=totals,
                by_formula={m: x[qualifies].sum(axis=0).tolist() for m, x in per_context.items()},
                narrow_minus_random=score)


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    verified_result(runs / PRIOR, REVISION, "carry_composes")
    aligned = verified_result(runs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    path = checkpoint_files(*HOST)
    donor_path = model_files("donor")
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    _, screen = crossed_panel(tok, 64, 130000)
    original, panel = crossed_panel(tok, 256, 131000)
    donor_tok = Tokenizer.from_file(str(donor_path / "tokenizer.json"))
    sources = {name: programs(donor_tok,len(original),index,computed=False,values_by_pair=[r[name] for r in original])
               for name,index in (("left",132000),("right",133000))}
    mapping, digest = archived_maps(runs / MAP_JOB)
    assert digest == aligned["map_digest"]
    axes, _, _, _, affine, bias = mapping
    targets = mx.repeat(gather_targets(donor_path, {"donor":sources}, original, axes, affine, bias), 2, axis=0)
    mx.eval(targets)
    model, _ = load_model(path, strict=True)
    clear = predict(model, screen, cut=CUT)
    endings = finish_tokens(tok, np.unique(clear["native"][..., 0]))
    threshold = int(count(outcomes(screen, clear, endings))["native"].all(axis=1).sum())
    print(f"Native paired-operator qualification: {threshold}/64", flush=True)
    np.savez_compressed(out / "screen.npz", terminators=np.array(sorted(endings)), **clear)
    report = dict(decision="benchmark_below_floor", screen_eligible=threshold, screen_groups=64,
                  models={"donor":MODELS["donor"],"recipient":HOST}, cut=CUT)
    if threshold >= 32:
        prior = runs / ALIGNMENT_JOB / "alignment.npz"
        keys = ("write", "read", "random", "center_input", "center_target")
        with np.load(prior) as data:
            geometry = tuple(mx.array(data[k].astype(np.float32)) for k in keys)
        dosing = {m: [] for m in ("both","random")}

        def intervene(state, start, length, mode):
            desired = targets[start:start+length].reshape(len(state), 2, -1)
            change = edits(state, desired, *geometry)[int(mode == "random")]
            dosing[mode].append(np.array(mx.linalg.norm(change, axis=-1)))
            return (state + change).transpose(1, 0, 2)

        scores = predict(model, panel, cut=CUT, operand_edit=intervene)
        doses = {m:np.concatenate(v) for m,v in dosing.items()}
        np.testing.assert_allclose(doses["both"],doses["random"],rtol=1e-5,atol=1e-6)
        endings = finish_tokens(tok, np.unique(np.concatenate([a[..., 0].ravel() for a in scores.values()])))
        np.savez_compressed(out / "operator.npz", **scores, terminators=np.array(sorted(endings)),
                            both_norm=doses["both"], random_norm=doses["random"])
        (out / "programs.json").write_text(json.dumps(dict(screen=screen,host=panel,donor=sources),
                                                       separators=(",",":"))+"\n")
        report.update(groups=len(original), map_digest=digest, alignment_digest=sha256(prior.read_bytes()).hexdigest(),
                      **summarize(outcomes(panel, scores, endings)))
    report.update(runtime_s=time.perf_counter()-start, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
