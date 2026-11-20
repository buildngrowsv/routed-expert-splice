"""Fit a small receiver-side connection for a verified frozen coding expert."""
import argparse
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval, read_logits
from large_graft import HOST_BLOCK, encode, finish, inputs
from large_opener import MODELS, model_files
from opener_programs import programs

WIDTH = 32
GROUPS_PER_BATCH = 4
EPOCHS = 4
LEARNING_RATE = 0.002
ARMS = ("host_feature", "shuffled_donor", "donor")


@dataclass
class Batch:
    state: mx.array
    position: mx.array
    choices: mx.array
    gold: mx.array
    host_feature: np.ndarray
    donor_feature: np.ndarray
    donor_logits: np.ndarray


def collect(host, coder, rows, tokens):
    batches = []
    for start in range(0, len(rows), 3 * GROUPS_PER_BATCH):
        sample = rows[start:start + 3 * GROUPS_PER_BATCH]
        ids, pos, choices, gold = encode(sample, tokens["host"])
        dids, dpos, dchoices, _ = encode(sample, tokens["donor"])
        np.testing.assert_array_equal(ids, dids)
        np.testing.assert_array_equal(np.asarray(pos), np.asarray(dpos))
        np.testing.assert_array_equal(np.asarray(choices), np.asarray(dchoices))
        ids = mx.array(ids)
        last = mx.arange(len(ids)), pos
        r, u = inputs(host, ids, HOST_BLOCK)
        native = host.model.layers[HOST_BLOCK].mlp(u)
        dr, du = inputs(coder, ids, len(coder.model.layers) - 1)
        expert = coder.model.layers[-1].mlp(du[last])
        donor_logits = read_logits(coder, coder.model.norm(dr[last] + expert), choices)
        state = r + native
        mx.eval(state)
        if not batches:
            expected = np.asarray(host.lm_head(host.model(ids)[last])[:, choices].astype(mx.float32))
            np.testing.assert_array_equal(expected.argmax(1), np.asarray(finish(host, state, pos, choices)).argmax(1))
            expected = read_logits(coder, coder.model(ids)[last], choices)
            np.testing.assert_array_equal(expected.argmax(1), donor_logits.argmax(1))
        batches.append(Batch(state, pos, choices, gold, np.asarray(native[last].astype(mx.float32)),
                             np.asarray(expert.astype(mx.float32)), donor_logits))
        if len(batches) % 16 == 0:
            print(f"Cached {start + len(sample)}/{len(rows)} prompt states", flush=True)
    return batches


def calibrate(train, name, width):
    values = np.concatenate([getattr(batch, name) for batch in train])
    triplets = values.reshape(-1, 3, values.shape[-1])
    contrast = (triplets - triplets.mean(axis=1, keepdims=True)).reshape(values.shape)
    _, _, right = np.linalg.svd(contrast.astype(np.float64), full_matrices=False)
    directions = right[:width].T.astype(np.float32)
    center = values.mean(axis=0)
    scale = ((values - center) @ directions).std(axis=0).clip(min=1e-4)
    return dict(center=center, directions=directions, scale=scale)


def project(batches, name, basis):
    return [mx.array((getattr(b, name) - basis["center"]) @ basis["directions"] / basis["scale"])
            for b in batches]


def score(host, batch, feature, weights):
    if weights is None:
        state = batch.state
    else:
        change = feature @ weights["port"] + weights["bias"]
        ix = mx.arange(len(batch.gold))
        state = batch.state.at[ix, batch.position].add(change.astype(batch.state.dtype))
    return finish(host, state, batch.position, batch.choices)


def fit(host, batches, features, arm, epochs, width):
    dimension = batches[0].state.shape[-1]
    weights = {"port": mx.zeros((width, dimension)), "bias": mx.zeros((dimension,))}
    optimizer = optim.Adam(LEARNING_RATE, bias_correction=True)
    order_rng = np.random.default_rng(41581)
    shuffle_rng = np.random.default_rng(910142)
    losses = []

    def objective(parameters, batch, source):
        return nn.losses.cross_entropy(score(host, batch, source, parameters), batch.gold, reduction="mean")

    step = mx.value_and_grad(objective)
    for epoch in range(epochs):
        for i in order_rng.permutation(len(batches)):
            batch, source = batches[i], features[i]
            if arm == "shuffled_donor":
                source = source[mx.array(shuffle_rng.permutation(len(source)))]
            loss, gradients = step(weights, batch, source)
            weights = optimizer.apply_gradients(gradients, weights)
            mx.eval(loss, weights, optimizer.state)
            assert np.isfinite(loss.item())
            losses.append(loss.item())
        print(f"{arm} epoch {epoch + 1}/{epochs} mean loss {np.mean(losses[-len(batches):]):.4f}", flush=True)
    return weights, losses


def panel(host, batches, features, weights, mode="aligned"):
    parts = []
    rng = np.random.default_rng(955591)
    for batch, source in zip(batches, features, strict=True):
        if mode == "off":
            source = mx.zeros_like(source)
        elif mode == "shuffled":
            source = source[mx.array(rng.permutation(len(source)))]
        elif mode == "swap":
            source = source.reshape(-1, 3, source.shape[-1])[:, [1, 2, 0], :].reshape(source.shape)
        parts.append(np.asarray(score(host, batch, source, weights)))
    return np.concatenate(parts).reshape(-1, 3, 4)


def summarize(logits, rows):
    size = len(rows) // 3
    assert all(v.shape == (size, 3, 4) and np.isfinite(v).all() for v in logits.values())
    choices = {k: v.argmax(axis=-1) for k, v in logits.items()}
    correct = {k: (v == np.arange(3)).all(axis=1) for k, v in choices.items()}
    offers = correct["intact_donor"] & ~correct["native"]
    new = correct["donor"] & ~correct["host_feature"] & ~correct["shuffled_donor"] & ~correct["native"]
    attributable = new & ~correct["off"]
    attributed_on_opportunities = attributable & offers
    mirrors = (choices["swap"] == np.array([1, 2, 0])).all(axis=1)
    draws = np.random.default_rng(283791).integers(size, size=(5000, size))
    contrasts = {k: interval(correct["donor"].astype(int) - correct[k], draws)
                 for k in ("host_feature", "shuffled_donor", "off")}
    passed = (offers.sum() >= 25 and attributed_on_opportunities.sum() >= 10
              and all(v["mean"] >= .05 and v["interval"][0] > 0 for v in contrasts.values()))
    ids = np.array([r["id"].rsplit("-", 1)[0] for r in rows[::3]])
    return dict(decision="learned_port_transfers_repairs" if passed else "unresolved",
                groups=size, complete={k: int(v.sum()) for k, v in correct.items()},
                donor_only_opportunities=ids[offers].tolist(),
                attributed_new_repairs=ids[attributable].tolist(),
                attributed_on_intact_donor_opportunities=ids[attributed_on_opportunities].tolist(),
                new_repairs_following_cyclic_donor=ids[attributable & mirrors].tolist(),
                donor_minus=contrasts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    training = programs(4 if smoke else 64, start=80000)
    test = programs(4 if smoke else 256, start=81000)
    assert {r["prompt"] for r in training}.isdisjoint(r["prompt"] for r in test)
    if not smoke:
        runs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
        previous = json.loads((runs / "large-qwen-donor-output-boundary" / "programs.json").read_text())
        assert {r["prompt"] for r in previous}.isdisjoint(r["prompt"] for r in training + test)
    paths = {k: model_files(k) for k in MODELS}
    tokens = {k: Tokenizer.from_file(str(v / "tokenizer.json")) for k, v in paths.items()}
    host, _ = load_model(paths["host"], strict=True)
    coder, _ = load_model(paths["donor"], strict=True)
    train = collect(host, coder, training, tokens)
    final = collect(host, coder, test, tokens)
    del coder
    gc.collect()
    width, epochs = (4, 1) if smoke else (WIDTH, EPOCHS)
    basis = {k: calibrate(train, k, width) for k in ("host_feature", "donor_feature")}
    features_train = {k: project(train, k, b) for k, b in basis.items()}
    features_final = {k: project(final, k, b) for k, b in basis.items()}
    source_final = features_final["donor_feature"]
    logits = {"native": panel(host, final, source_final, None),
              "intact_donor": np.concatenate([b.donor_logits for b in final]).reshape(-1, 3, 4)}
    fitted, history = {}, {}
    for arm in ARMS:
        source = "host_feature" if arm == "host_feature" else "donor_feature"
        fitted[arm], history[arm] = fit(host, train, features_train[source], arm, epochs, width)
        logits[arm] = panel(host, final, features_final[source], fitted[arm],
                            mode="shuffled" if arm == "shuffled_donor" else "aligned")
    for mode in ("off", "swap"):
        logits[mode] = panel(host, final, source_final, fitted["donor"], mode=mode)
    summary = summarize(logits, test)
    summary.update(models=MODELS, phase="smoke" if smoke else "full", rank=width,
                   epochs=epochs, groups_training=len(training)//3,
                   insert_block_one_based=HOST_BLOCK + 1,
                   trainable_parameters=sum(v.size for v in fitted["donor"].values()),
                   step_lr=LEARNING_RATE, seconds=time.perf_counter() - started,
                   peak_mlx_bytes=mx.get_peak_memory())
    if smoke:
        summary["decision"] = "smoke_complete"
        print(json.dumps(summary))
        if "STUDIO_JOB_DIR" in os.environ:
            (Path(os.environ["STUDIO_JOB_DIR"]) / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    out = Path(os.environ["STUDIO_JOB_DIR"])
    archive = {f"{kind}_{arm}_{name}": mx.array(value)
               for kind, arms in (("basis", basis), ("learner", fitted))
               for arm, data in arms.items() for name, value in data.items()}
    weights = out / "ports.safetensors"
    mx.save_safetensors(str(weights), archive)
    summary["port_sha256"] = sha256(weights.read_bytes()).hexdigest()
    np.savez_compressed(out / "scores.npz", **logits)
    (out / "programs.json").write_text(json.dumps(dict(train=training, test=test), indent=2) + "\n")
    (out / "training.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
