"""Test whether the same whole-state key edit gains when the key assignment moves."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import encoded_batches, prefix
from identifier_binding import patch
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import CUT, HOST
from identifier_numeric_open import whole_vocabulary
from large_opener import checkpoint_files

EARLY_JOB = "qwen-indexed-address-72b"
EARLY_REVISION = "ddcca579733367d0282894afd3dc752b47a74bc0"


def move_key(tokenizer, rows):
    result = []
    for row in rows:
        prompts = []
        for original, expected in zip(row["prompts"], row["answers"], strict=True):
            head, key, table, last = original.split("\n", 3)
            assert head == "def fetch():" and key.startswith("    mode_") and table.startswith("    table = ")
            changed = "\n".join((head, table, key, last))
            exec(compile(changed + expected + "\n", "<moved key lookup>", "exec"), {})
            prompts.append(changed)
        ids = [tokenizer.encode(text, add_special_tokens=False).ids for text in prompts]
        site, = [i for i, (x, y) in enumerate(zip(*ids, strict=True)) if x != y]
        assert [tokenizer.decode([seq[site]]).strip() for seq in ids] == list(map(str, row["selector"]))
        completions = [tokenizer.encode(p + a, add_special_tokens=False).ids
                       for p, a in zip(prompts, row["answers"], strict=True)]
        assert [x[:-1] for x in completions] == ids
        result.append(dict(row, ids=ids, prompts=prompts, site=site, choices=[x[-1] for x in completions]))
    return result


def evaluate(model, rows, *, cut=CUT):
    native, whole = [], []
    for i, (ids, site, last, candidates, _) in enumerate(encoded_batches(rows), start=1):
        state = prefix(model, ids, cut)
        j = mx.arange(len(ids))
        native.extend(whole_vocabulary(model, state, last, cut, candidates))
        partner = j ^ 1
        exchanged = patch(state, site, state[partner, site])
        whole.extend(whole_vocabulary(model, exchanged, last, cut, candidates))
        if i % 12 == 0:
            print(f"Scored {i * 8}/{len(rows)} contexts", flush=True)
    return dict(native=np.asarray(native), whole=np.asarray(whole))


def summarize(early, late):
    native_early = (early["native"][..., 0] == 1).reshape(-1, 6).all(axis=1)
    native_late = (late["native"][..., 0] == 1).reshape(-1, 6).all(axis=1)
    keep = native_early & native_late
    n = int(keep.sum())
    paired = None
    count = {}
    switched = {}
    for name, source in (("early", early), ("late", late)):
        values = source["whole"].reshape(-1, 3, 2, 4)[keep]
        resolved = (values[:, :2, :, 1] == 1).all(axis=(1, 2))
        inert = (values[:, 2, :, 0] == 1).all(axis=1)
        switched[name] = resolved
        count[name] = dict(switch_both=int(resolved.sum()), inert=int(inert.sum()),
                           switch_and_preserve=int((resolved & inert).sum()))
    if n:
        draws = np.random.default_rng(325140).integers(n, size=(5000, n))
        paired = interval(switched["late"].astype(int) - switched["early"], draws)
    passed = (n >= 100 and count["late"]["switch_both"] >= .8 * n
              and count["late"]["inert"] >= .95 * n and paired["mean"] >= .15 and paired["interval"][0] > 0)
    return dict(decision="key_order_restores_whole_state_routing" if passed else "unresolved",
                shared_native_groups=n, native_early=int(native_early.sum()),
                native_late=int(native_late.sum()), whole=count, late_minus_early=paired)


def main():
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    source = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs" / EARLY_JOB
    prior = verified_result(source, EARLY_REVISION, "unresolved")
    payload = (source / "programs.json").read_bytes()
    assert sha256(payload).hexdigest() == prior["programs_digest"]
    panel = json.loads(payload)["host"]
    checkpoint = checkpoint_files(*HOST)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rows = move_key(tokenizer, panel)
    model, _ = load_model(checkpoint, strict=True)
    late = evaluate(model, rows)
    raw = source / "selector.npz"
    with np.load(raw) as old:
        early = {m: old[m] for m in ("native", "whole")}
    assert all(z.shape == (768, 2, 4) for z in (*early.values(), *late.values()))
    np.savez_compressed(out / "moved-key.npz", **late)
    (out / "programs.json").write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    report = dict(model=HOST, cut=CUT, original_programs_digest=prior["programs_digest"],
                  original_ranks_digest=sha256(raw.read_bytes()).hexdigest(),
                  runtime_s=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory(),
                  **summarize(early, late))
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
