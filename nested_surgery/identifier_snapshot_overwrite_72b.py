"""Compare a narrow donor edit and an overwrite-only full-state exchange."""
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import re
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES
from identifier_computation import programs
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, HOST
from identifier_numeric_route_72b import MODES, evaluate
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_numeric_select_72b import panel
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files


def crossed(tokenizer, count, start):
    original, _ = panel(tokenizer, count, start, snapshot=True)
    rows = []
    values = []
    for saved, current in zip(original[::2], original[1::2], strict=True):
        _, new_values = saved["sources"]
        offset = saved["offset"]
        answers = [str(d + offset) for d in new_values]
        values.append(new_values)
        for old in range(2):
            for row in (saved, current):
                name = f"left_{row['id']}"
                prefix = row["prompts"][old]
                pattern = re.compile(rf"(?m)^(    {name} = )\d+$")
                hits = list(pattern.finditer(prefix))
                assert len(hits) == 2
                point = hits[-1]
                prompts = [prefix[:point.start()] + point[1] + str(d) + prefix[point.end():]
                           for d in new_values]
                tokens = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
                site, = [i for i, (a, b) in enumerate(zip(*tokens, strict=True)) if a != b]
                assert site == row["sites"][1]
                expected = answers if row["chosen"] else [saved["answers"][old]] * 2
                full = [tokenizer.encode(p + d, add_special_tokens=False).ids
                        for p, d in zip(prompts, expected, strict=True)]
                overwrite = [tokenizer.encode(p + d, add_special_tokens=False).ids
                             for p, d in zip(prompts, answers, strict=True)]
                assert [t[:-1] for t in full] == tokens == [t[:-1] for t in overwrite]
                for prompt, d in zip(prompts, expected, strict=True):
                    exec(compile(prompt + d + "\n", "<crossed saved value>", "exec"), {})
                rows.append(dict(id=row["id"], old_variant=old, chosen=row["chosen"], ids=tokens,
                                 prompts=prompts, site=site, answers=expected,
                                 choices=[t[-1] for t in full], labels=list(map(int, expected)),
                                 overwrite=new_values, alternate_overwrite_choices=[t[-1] for t in overwrite[::-1]]))
    return rows, values


def native_complete(scores):
    return (scores[..., 0] == 1).reshape(-1, 2, 2, 2).all(axis=(1, 2, 3))


def summarize(rows, scores):
    keep = native_complete(scores["native"])
    n = int(keep.sum())
    alternatives = np.array([r["alternate_overwrite_choices"] for r in rows])
    switches, preservation, leak = {}, {}, {}
    differences = {}
    for mode in MODES[1:]:
        rank, alternate, top, _ = scores[mode].transpose(2, 0, 1)
        saved_ok = (rank == 1).reshape(-1, 2, 2, 2)[keep, :, 0]
        current_ok = (alternate == 1).reshape(-1, 2, 2, 2)[keep, :, 1]
        uses_overwrite = ((rank != 1) & (top == alternatives)).reshape(-1, 2, 2, 2)[keep, :, 0]
        switches[mode] = int(current_ok.all(axis=(1, 2)).sum())
        preservation[mode] = int(saved_ok.all(axis=(1, 2)).sum())
        leak[mode] = uses_overwrite.any(axis=(1, 2))
    decision = "unresolved"
    if n:
        draws = np.random.default_rng(847021).integers(n, size=(5000, n))
        pairs = (("donor", "whole"), ("donor", "orthogonal"), ("whole", "orthogonal"))
        for a, b in pairs:
            differences[f"{a}_minus_{b}"] = interval(leak[a].astype(int) - leak[b], draws)
        matched = differences["donor_minus_whole"]
        qualified = n >= 100 and min(switches[m] for m in ("donor", "whole")) >= .8 * n
        if qualified:
            if matched["mean"] >= .03 and matched["interval"][0] > 0:
                decision = "donor_leaks_more"
            elif matched["mean"] <= -.03 and matched["interval"][1] < 0:
                decision = "whole_leaks_more"
            elif (matched["interval"][0] >= -.03 and matched["interval"][1] <= .03
                  and min(differences[k]["interval"][0] for k in ("donor_minus_orthogonal",
                                                                  "whole_minus_orthogonal")) > 0):
                decision = "shared_overwrite_interference"
    return dict(decision=decision, eligible=n, current_switches=switches, saved_preserved=preservation,
                overwritten_answer={m: int(v.sum()) for m, v in leak.items()}, paired=differences)


def main():
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    prior = verified_result(jobs / ALIGNMENT_JOB, ALIGNMENT_REVISION, "cross_width_transfer")
    recipient = checkpoint_files(*HOST)
    tokenizer = Tokenizer.from_file(str(recipient / "tokenizer.json"))
    pilot, _ = crossed(tokenizer, 64, 171000)
    model, _ = load_model(recipient, strict=True)
    scores, _ = evaluate(model, pilot)
    n = int(native_complete(scores["native"]).sum())
    print(f"Native crossed-overwrite qualification: {n}/64", flush=True)
    np.savez_compressed(out / "screen.npz", **scores)
    (out / "screen.json").write_text(json.dumps(pilot, separators=(",", ":")) + "\n")
    report = dict(decision="benchmark_below_floor", screen_eligible=n, screen_groups=64,
                  models={"donor": MODELS["donor"], "recipient": HOST})
    if n >= 40:
        del model
        gc.collect()
        rows, values = crossed(tokenizer, 256, 172000)
        mapping, digest = archived_maps(jobs / MAP_JOB)
        assert digest == prior["map_digest"]
        donor_axis, _, _, _, affine, bias = mapping
        path = model_files("donor")
        donor_tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        source = programs(donor_tok, len(values), 173000, computed=False, values_by_pair=values)
        donor, _ = load_model(path, strict=True)
        states = mx.array(gather(donor, source, SITES["donor"]).astype(np.float32))
        aligned = (states @ donor_axis @ affine + bias).reshape(len(source), 2, -1)
        target = mx.repeat(aligned, 4, axis=0).reshape(len(rows) * 2, -1)
        mx.eval(target)
        del donor
        gc.collect()
        alignment = jobs / ALIGNMENT_JOB / "alignment.npz"
        keys = ("write", "read", "random", "center_input", "center_target")
        with np.load(alignment) as z:
            geometry = tuple(mx.array(z[k].astype(np.float32)) for k in keys)
        assert geometry[0].shape == (8192, 7)
        model, _ = load_model(recipient, strict=True)
        scores, norms = evaluate(model, rows, target, geometry)
        np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
        np.savez_compressed(out / "overwrite.npz", norms=norms, **scores)
        (out / "programs.json").write_text(json.dumps(dict(host=rows, donor=source), separators=(",", ":")) + "\n")
        report.update(map_digest=digest, alignment_digest=sha256(alignment.read_bytes()).hexdigest(),
                      groups=len(values), **summarize(rows, scores))
    report.update(runtime_s=time.perf_counter() - started, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
