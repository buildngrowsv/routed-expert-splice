"""Test a frozen operand interface on conditional and loop completions."""

import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, prefix
from identifier_binding import patch
from identifier_computation import DIGITS, programs
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, edits
from identifier_numeric_scale import PRIOR, archived_maps
from identifier_numeric_width_72b import HOST
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files


FAMILIES = ("conditional", "loop")
ARMS = ("native", "bridge", "wrong", "orthogonal", "dense", "whole")
SUFFIX = "\nassert fetch() == ("
TUPLE_END = re.compile(r"^\)[ \t]*(?:$|[\n\r#])")
BUDGET_SECONDS = 3 * 60 * 60


def make_group(tok, family, index):
    rng = random.Random(f"flagship-transfer-v1-{family}-{index}")
    name = f"reading_{index}"
    checksum = rng.randrange(50, 90)
    if family == "conditional":
        lower, upper = rng.choice(((2, 4), (2, 5), (2, 6), (3, 5), (3, 6), (4, 6)))
        values = [rng.randint(1, lower), rng.randint(lower + 1, upper), rng.randint(upper + 1, 8)]
        rng.shuffle(values)
        codes = rng.sample(range(11, 50), 3)
        tail = (f"    if {name} < {lower}.5:\n        result = {codes[0]}\n"
                f"    elif {name} < {upper}.5:\n        result = {codes[1]}\n"
                f"    else:\n        result = {codes[2]}\n")
        expected = [codes[sorted(values).index(v)] for v in values]
    else:
        values = rng.sample(DIGITS, 3)
        offset, step = rng.randrange(12, 22), rng.randrange(1, 5)
        tail = (f"    result = {offset}\n    for _ in range({name}):\n"
                f"        result += {step}\n")
        expected = [offset + step * v for v in values]
    tail = tail + f"    return result, {checksum}\n"
    text = [f"def fetch():\n    {name} = {v}\n" + tail + SUFFIX for v in values[:2]]
    ids = [tok.encode(s, add_special_tokens=False).ids for s in text]
    site, = (i for i, pair in enumerate(zip(*ids, strict=True)) if pair[0] != pair[1])
    assert [tok.decode([row[site]]).strip() for row in ids] == [str(v) for v in values[:2]]
    answers = [str((result, checksum)) for result in expected]
    assert len(set(answers)) == 3
    for value, answer in zip(values, answers, strict=True):
        program = f"def fetch():\n    {name} = {value}\n" + tail
        exec(compile(program + SUFFIX + answer[1:], "<generated flagship benchmark>", "exec"), {})
    suffixes = None
    for prompt, encoded in zip(text, ids, strict=True):
        completions = [tok.encode(prompt + answer[1:-1], add_special_tokens=False).ids for answer in answers[:2]]
        assert all(seq[:len(encoded)] == encoded for seq in completions)
        tokens = [seq[len(encoded):] for seq in completions]
        assert all(tok.decode(bits, skip_special_tokens=False) == answer[1:-1]
                   for bits, answer in zip(tokens, answers[:2], strict=True))
        assert suffixes is None or suffixes == tokens
        suffixes = tokens
    assert suffixes[0] != suffixes[1]
    return dict(id=index, values=values, site=site, prompts=text, ids=ids, answers=answers, tokens=suffixes)


def panel(tok, count, start):
    return {family: [make_group(tok, family, i) for i in range(start, start + count)]
            for family in FAMILIES}


def dense_projection(calibration, target_deltas, donor_deltas):
    differences = calibration[1::2] - calibration[::2]
    kernel = differences @ differences.T
    penalty = .02 * np.mean(np.diag(kernel))
    alpha = np.linalg.solve(kernel + penalty * np.eye(len(kernel)), target_deltas)
    return (donor_deltas @ differences.T @ alpha).astype(np.float32)


def donor_states(tok, model, rows, start):
    site = SITES["donor"]
    pairs = [r["values"][:2] for r in rows]
    wrong = [[r["values"][2], r["values"][0]] for r in rows]
    ordinary = programs(tok, len(rows), start, computed=False, values_by_pair=pairs)
    decoy = programs(tok, len(rows), start + 10000, computed=False, values_by_pair=wrong)
    return (gather(model, ordinary, site).reshape(len(rows), 2, -1),
            gather(model, decoy, site)[::2])


def predict(model, tokens, sites, arm, geometry, payload):
    max_tokens = max(len(seq) for seq in tokens)
    h = prefix(model, mx.array([seq + [0] * (max_tokens - len(seq)) for seq in tokens]), CUT)
    norms = None
    if arm != "native":
        state = h[mx.arange(len(tokens)), sites].astype(mx.float32)
        targets, wrong, dense = payload
        delta, orthogonal = edits(state, targets, *geometry)
        if arm == "whole":
            change = state[mx.arange(len(tokens)) ^ 1] - state
        elif arm == "dense":
            ratio = mx.linalg.norm(delta, axis=-1, keepdims=True) / mx.maximum(
                mx.linalg.norm(dense, axis=-1, keepdims=True), 1e-12)
            change = dense * ratio
        elif arm == "wrong":
            change, _ = edits(state, wrong, *geometry)
        else:
            change = orthogonal if arm == "orthogonal" else delta
        norms = np.array(mx.linalg.norm(change, axis=-1))
        h = patch(h, sites, state + change)
    mask = create_attention_mask(h, None)
    for layer in model.model.layers[CUT:]:
        h = layer(h, mask)
    return h, norms


def score(model, tok, rows, arms, *, geometry=None, payload=None, deadline=None):
    logits_head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    hits = {arm: [] for arm in arms}
    traces = {arm: [] for arm in arms}
    norms = {arm: [] for arm in arms if arm != "native"}
    for start in range(0, len(rows), 8):
        if deadline is not None and time.monotonic() >= deadline:
            break
        batch = rows[start:start + 8]
        prefixes = [seq for row in batch for seq in row["ids"]]
        sites = mx.array(np.repeat([row["site"] for row in batch], 2))
        sources = None if payload is None else tuple(x[2 * start:2 * (start + len(batch))] for x in payload)
        for arm in arms:
            direction = (0, 1) if arm == "native" else (1, 0)
            suffixes = [row["tokens"][i] for row in batch for i in direction]
            sequences = [a + b for a, b in zip(prefixes, suffixes, strict=True)]
            h, dose = predict(model, sequences, sites, arm, geometry, sources)
            count = 1 + max(map(len, suffixes))
            steps = [[min(len(a) - 1 + j, len(s) - 1) for j in range(count)]
                     for a, s in zip(prefixes, sequences, strict=True)]
            states = model.model.norm(h[mx.arange(len(sequences))[:, None], mx.array(steps)])
            logits = logits_head(states).astype(mx.float32)
            maxima = mx.argmax(logits, axis=-1)
            unique = mx.sum(logits == logits.max(axis=-1, keepdims=True), axis=-1) == 1
            top, tied = np.array(maxima), ~np.array(unique)
            for target, candidates, ties in zip(suffixes, top, tied, strict=True):
                k = len(target)
                joined = np.stack((candidates[:k + 1], ~ties[:k + 1]), axis=-1)
                traces[arm].append(joined.tolist())
                close = tok.decode([int(candidates[k])], skip_special_tokens=False)
                hits[arm].append(candidates[:k].tolist() == target and not ties[:k + 1].any()
                                 and TUPLE_END.match(close) is not None)
            if dose is not None:
                norms[arm].extend(dose.tolist())
        if deadline is not None and (start + len(batch)) % 64 == 0:
            print(f"Scored {start + len(batch)}/{len(rows)} paired groups", flush=True)
    return ({a: np.asarray(hit, dtype=bool).reshape(-1, 2).all(1) for a, hit in hits.items()},
            dict(traces=traces, norms=norms))


def summarize(wins):
    qualified = wins["native"]
    n = int(qualified.sum())
    overall = {a: int(v.sum()) for a, v in wins.items()}
    matched = {a: int(v[qualified].sum()) for a, v in wins.items()}
    if not n:
        return dict(groups=len(qualified), qualified=n, total_success=overall, qualified_success=matched)
    rng = np.random.default_rng(833721)
    draw = rng.integers(n, size=(5000, n))
    margins = {a: interval((wins["bridge"].astype(int) - wins[a])[qualified], draw)
               for a in ("wrong", "orthogonal", "dense")}
    return dict(groups=len(qualified), qualified=n, total_success=overall, qualified_success=matched,
                bridge_minus=margins)


def main():
    start = time.monotonic()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    mx.set_cache_limit(2 * 1024**3)
    recipient_path = checkpoint_files(*HOST)
    recipient_tok = Tokenizer.from_file(str(recipient_path / "tokenizer.json"))
    screen = panel(recipient_tok, 64, 210000)
    host, _ = load_model(recipient_path, strict=True)
    assert len(host.model.layers) == 80 and host.args.hidden_size == 8192
    scores = {}
    for family, rows in screen.items():
        hits, ledger = score(host, recipient_tok, rows, ("native",))
        scores[family] = int(hits["native"].sum())
        (out / f"{family}-screen.json").write_text(json.dumps(dict(rows=rows, **ledger)))
    report = dict(decision="native_screen_below_floor", screen_qualified=scores,
                  models={"donor": MODELS["donor"], "recipient": HOST}, cuts=[SITES["donor"], CUT])
    print(f"Native screen, out of 64 per family: {scores}", flush=True)
    if min(scores.values()) >= 32:
        del host
        gc.collect()
        report.update(run_panel(recipient_tok, recipient_path, root, out))
    report.update(runtime_s=time.monotonic() - start, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def run_panel(tok, host_path, root, out):
    alignment = root / ALIGNMENT_JOB
    prior = verified_result(alignment, ALIGNMENT_REVISION, "cross_width_transfer")
    mapping, digest = archived_maps(root / PRIOR)
    assert prior["map_digest"] == digest
    donor_basis, _, _, _, affine, bias = mapping
    interface = alignment / "alignment.npz"
    with np.load(interface) as artifact:
        keys = ("write", "read", "random", "center_input", "center_target")
        geometry = tuple(mx.array(artifact[key].astype(np.float32)) for key in keys)
        calibration_target = artifact["y"]
    assert geometry[0].shape == (8192, 7)
    rows_by_family = panel(tok, 768, 220000)
    (out / "programs.json").write_text(json.dumps(rows_by_family, separators=(",", ":")))
    donor_path = model_files("donor")
    donor_tok = Tokenizer.from_file(str(donor_path / "tokenizer.json"))
    donor, _ = load_model(donor_path, strict=True)
    old_calibration = json.loads((alignment / "calibration.json").read_text())
    calibration = programs(donor_tok, 256, 125000, computed=False)
    assert [p["prompts"] for p in calibration] == [p["prompts"] for p in old_calibration]
    donor_cal = gather(donor, calibration, SITES["donor"])
    payload = {}
    for j, (family, rows) in enumerate(rows_by_family.items()):
        paired, wrong = donor_states(donor_tok, donor, rows, 260000 + j * 20000)
        desired = mx.array(paired[:, ::-1].reshape(2 * len(rows), -1).astype(np.float32))
        decoy = mx.array(np.repeat(wrong, 2, axis=0).astype(np.float32))
        dense = dense_projection(donor_cal, calibration_target, (paired[:, ::-1] - paired).reshape(2 * len(rows), -1))
        payload[family] = (desired @ donor_basis @ affine + bias, decoy @ donor_basis @ affine + bias,
                           mx.array(dense))
        mx.eval(*payload[family])
    del donor
    gc.collect()
    host, _ = load_model(host_path, strict=True)
    deadline = time.monotonic() + BUDGET_SECONDS
    comparisons = {}
    for family, rows in rows_by_family.items():
        print(f"Testing {family}", flush=True)
        hits, ledger = score(host, tok, rows, ARMS, geometry=geometry, payload=payload[family], deadline=deadline)
        doses = ledger["norms"]
        for matched_arm in ("orthogonal", "dense"):
            np.testing.assert_allclose(doses["bridge"], doses[matched_arm], rtol=1e-5, atol=1e-6)
        (out / f"{family}-scores.json").write_text(json.dumps(ledger, separators=(",", ":")))
        comparisons[family] = summarize(hits)
    complete = all(row["groups"] == 768 for row in comparisons.values())
    controls_pass = complete and all(row["qualified"] >= 384 and
                                    row["qualified_success"]["whole"] >= .7 * row["qualified"]
                                    for row in comparisons.values())
    passed = controls_pass and all(row["qualified_success"]["bridge"] >= .7 * row["qualified"] and
                                  all(row["bridge_minus"][arm]["mean"] >= .2 and
                                      row["bridge_minus"][arm]["interval"][0] > 0
                                      for arm in ("wrong", "orthogonal")) for row in comparisons.values())
    beats_dense = passed and all(row["bridge_minus"]["dense"]["mean"] >= .05 and
                                 row["bridge_minus"]["dense"]["interval"][0] > 0
                                 for row in comparisons.values())
    if not complete:
        decision = "scoring_budget_exhausted"
    elif not controls_pass:
        decision = "positive_control_below_floor"
    else:
        decision = "exact_transfer_composes" if passed else "exact_transfer_below_floor"
    return dict(decision=decision, comparisons=comparisons, beats_dense=beats_dense, map_sha256=digest,
                alignment_sha256=sha256(interface.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
