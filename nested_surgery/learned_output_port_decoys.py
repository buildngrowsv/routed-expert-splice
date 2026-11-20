"""Compare the frozen repair bridges on new code containing bracket decoys."""
import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from large_opener import MODELS, PATCHES, model_files
from learned_output_port import ARMS, collect, panel, project, summarize
from opener_programs import programs

PRIOR = "large-qwen-learned-output-port"
REVISION = "a733c8662949bf34ce8c86f7d4fb17f5a945c014"


def rewrite(rows, decoy):
    rewritten = []
    for row in rows:
        group = row["id"].rsplit("-", 1)[0]
        rng = random.Random(f"repair-decoy-v1-{group}")
        text = "".join(rng.choices("()[]{}", k=12))
        if not decoy:
            text = text.translate(str.maketrans("()[]{}", "abcdef"))
        head, tail = row["prompt"].split("<|fim_suffix|>")
        stem, assignment = head.removeprefix("<|fim_prefix|>").split("    value = ")
        opener, expression = assignment[0], assignment[1:]
        new = f"{stem}    value = {opener}({expression}, '{text[:6]}')[0]  # {text[6:]}\n        "
        source_tail = tail.removesuffix("<|fim_middle|>")
        valid = []
        for i, candidate in enumerate(PATCHES):
            try:
                tree = ast.parse(new + candidate + source_tail)
            except SyntaxError:
                continue
            env = {"__builtins__": {"sum": sum}}
            exec(compile(tree, "<matched bracket decoy>", "exec"), env)
            assert env["transform"](-23) == -23
            valid.append(i)
        assert valid == [row["gold"]]
        rewritten.append(dict(row, prompt=f"<|fim_prefix|>{new}<|fim_suffix|>{tail}"))
    return rewritten


def endpoints(host, coder, rows, tokens, basis, weights):
    batches = collect(host, coder, rows, tokens)
    sources = {name: project(batches, name, fit) for name, fit in basis.items()}
    donor = sources["donor_feature"]
    ranks = {"native": panel(host, batches, donor, None),
             "intact_donor": np.concatenate([b.donor_logits for b in batches]).reshape(-1, 3, 4)}
    for arm in ARMS:
        source = sources["host_feature"] if arm == "host_feature" else donor
        mode = "shuffled" if arm == "shuffled_donor" else "aligned"
        ranks[arm] = panel(host, batches, source, weights[arm], mode=mode)
    for mode in ("off", "swap"):
        ranks[mode] = panel(host, batches, donor, weights["donor"], mode=mode)
    return ranks


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--check-generator", action="store_true")
    opt = cli.parse_args()
    rows = programs(256, start=83000)
    panels = {"neutral": rewrite(rows, False), "decoys": rewrite(rows, True)}
    if opt.check_generator:
        assert all(len({r["prompt"] for r in f}) == 768 for f in panels.values())
        print("Verified 512 matched executable repair triplets")
        return
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    prior = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs" / PRIOR
    receipt = json.loads((prior / "receipt.json").read_text())
    assert (receipt["revision"], receipt["status"], receipt["exit"]) == (REVISION, "done", 0)
    old = json.loads((prior / "result.json").read_text())
    assert old["complete"]["host_feature"] == 253 and old["decision"] == "unresolved"
    path = prior / "ports.safetensors"
    assert sha256(path.read_bytes()).hexdigest() == old["port_sha256"]
    archive = mx.load(str(path))
    basis = {arm: {k: np.asarray(archive[f"basis_{arm}_{k}"]) for k in ("center", "directions", "scale")}
             for arm in ("host_feature", "donor_feature")}
    weights = {arm: {k: archive[f"learner_{arm}_{k}"] for k in ("port", "bias")} for arm in ARMS}
    models = {k: model_files(k) for k in MODELS}
    tokens = {k: Tokenizer.from_file(str(p / "tokenizer.json")) for k, p in models.items()}
    host, _ = load_model(models["host"], strict=True)
    coder, _ = load_model(models["donor"], strict=True)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    results = {}
    for name, rows in panels.items():
        print(f"Scoring {name} layout with all bridges frozen", flush=True)
        values = endpoints(host, coder, rows, tokens, basis, weights)
        np.savez_compressed(out / f"{name}.npz", **values)
        results[name] = summarize(values, rows)
    decoys, neutral = results["decoys"], results["neutral"]
    headroom = neutral["complete"]["host_feature"] - decoys["complete"]["host_feature"]
    decision = "unresolved"
    if headroom < 26:
        decision = "decoy_shift_lacks_headroom"
    elif decoys["decision"] == "learned_port_transfers_repairs":
        decision = "frozen_donor_adds_decoy_repairs"
    report = dict(decision=decision, endpoints=results, host_groups_lost=headroom,
                  port_sha256=old["port_sha256"], models=MODELS,
                  seconds=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory())
    (out / "programs.json").write_text(json.dumps(panels, separators=(",", ":")) + "\n")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
