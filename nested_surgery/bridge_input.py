"""Isolate the frozen input map with all within-group source and destination pairings."""
from hashlib import file_digest
from itertools import combinations
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from large_graft import HOST_BLOCK, encode, inputs
from large_opener import MODELS, model_files
from opener_programs import programs


def signal(logits):
    # Dimensions: group, destination, source, candidate.
    pieces = []
    for a, b in combinations(range(3), 2):
        contrast = logits[..., a] - logits[..., b]
        pieces.append((contrast[:, :, a] - contrast[:, :, b]).mean(axis=1))
    return np.stack(pieces, axis=1)


def interval(values, idx):
    return dict(mean=float(values.mean()), interval=np.quantile(values[idx].mean(1), [.025, .975]).tolist())


def read_logits(model, hidden, choices):
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    values = [np.array(head(hidden[i:i + 8])[:, choices].astype(mx.float32)) for i in range(0, len(hidden), 8)]
    return np.concatenate(values)


def main():
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    location = runs / "large-qwen-block27-graft/interfaces.safetensors"
    maps = mx.load(str(location))
    w, bias = maps["coder_w"], maps["coder_b"]
    del maps
    with location.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    rows = programs(256, start=40000)
    previous = json.loads((runs / "large-qwen-block27-graft/programs.json").read_text())
    assert {x["prompt"] for x in rows}.isdisjoint(x["prompt"] for x in previous)
    paths = {key: model_files(key) for key in ("donor","host")}
    tokens = {key: Tokenizer.from_file(str(path / "tokenizer.json")) for key, path in paths.items()}
    donor, _ = load_model(paths["donor"], strict=True)
    host, _ = load_model(paths["host"], strict=True)
    expert = donor.model.layers[-1].mlp
    residuals, updates = [], {"native": [], "mapped": []}
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        ids, pos, choices, _ = encode(batch, tokens["donor"])
        hids, hpos, hchoices, _ = encode(batch, tokens["host"])
        np.testing.assert_array_equal(ids, hids)
        np.testing.assert_array_equal(np.array(pos), np.array(hpos))
        np.testing.assert_array_equal(np.array(choices), np.array(hchoices))
        last = (mx.arange(len(batch)), pos)
        residual, u = inputs(donor, mx.array(ids), len(donor.model.layers)-1)
        _, host_u = inputs(host, mx.array(hids), HOST_BLOCK)
        mapped = (host_u.astype(mx.float32) @ w + bias).astype(mx.bfloat16)
        residuals.append(residual[last])
        updates["native"].append(expert(u)[last])
        updates["mapped"].append(expert(mapped)[last])
        mx.eval(residuals[-1], updates["native"][-1], updates["mapped"][-1])
        if i == 0:
            direct = read_logits(donor, donor.model(mx.array(ids))[last], choices)
    r = mx.concatenate(residuals)
    groups = np.arange(len(rows)).reshape(-1, 3)
    gold = np.array([x["gold"] for x in rows])
    np.testing.assert_array_equal(gold, np.tile(np.arange(3), len(groups)))
    cubes, effects, complete = {}, {}, {}
    for key, parts in updates.items():
        v = mx.concatenate(parts)
        cells = []
        for src in range(3):
            which = mx.array(np.repeat(groups[:, src], 3))
            hidden = donor.model.norm(r + v[which])
            cells.append(read_logits(donor, hidden, choices).reshape(-1, 3, 4))
        cube = np.stack(cells, axis=2)
        assert np.isfinite(cube).all()
        diagonal = cube[:, np.arange(3), np.arange(3), :].reshape(-1, 4)
        if key == "native":
            np.testing.assert_allclose(diagonal[:8], direct, atol=.25, rtol=.02)
            np.testing.assert_array_equal(diagonal[:8].argmax(1), direct.argmax(1))
        cubes[key] = cube
        effects[key] = signal(cube)
        complete[key] = int((diagonal.argmax(1) == gold).reshape(-1, 3).all(1).sum())
    off = read_logits(donor, donor.model.norm(r), choices)
    assert np.isfinite(off).all()
    complete["off"] = int((off.argmax(1) == gold).reshape(-1, 3).all(1).sum())
    idx = np.random.default_rng(445638).integers(len(groups), size=(5000, len(groups)))
    native = interval(effects["native"].mean(1), idx)
    mapped = interval(effects["mapped"].mean(1), idx)
    contrast = interval((effects["mapped"]-.5*effects["native"]).mean(1), idx)
    outcome = "unresolved"
    if native["interval"][0] > 0:
        if contrast["interval"][0] > 0:
            outcome = "retained"
        elif contrast["interval"][1] < 0:
            outcome = "attenuated"
    np.savez_compressed(out / "crossed-scores.npz", **cubes, off=off, gold=gold)
    (out / "programs.json").write_text(json.dumps(rows, indent=2)+"\n")
    report = dict(decision=outcome, native=native, mapped=mapped, mapped_minus_half_native=contrast,
                  pair_means={k: v.mean(0).tolist() for k,v in effects.items()}, complete=complete,
                  interface_sha256=digest, models=MODELS, full_forward_preflight=8)
    (out / "result.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
