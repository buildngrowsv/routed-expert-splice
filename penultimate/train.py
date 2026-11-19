"""Adapt all parameters of block 28, with FP32 master weights and BF16 forwards."""
import json
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from features import ARMS, P, RESULTS, WORK, load_model, modules, scored_output, tokens, write
from forced import SEEDS
from prepare import POS

CHECKS = (0, 8, 32, 96)


def bf16(weights):
    return tree_map(lambda x: x.astype(mx.bfloat16), weights)


def measure(host, block, x, y):
    losses, hits = [], []
    for i in range(0, len(x), 8):
        out = scored_output(block, x[i:i+8])
        logits = host.model.embed_tokens.as_linear(host.model.norm(out)).astype(mx.float32)
        labels = y[i:i+8]
        losses.append(np.array(nn.losses.cross_entropy(logits, labels, reduction="none")).ravel())
        hits.append(np.array(logits.argmax(-1) == labels).ravel())
    return np.concatenate(losses), np.concatenate(hits)


def live_scores(host, graft, ids):
    layer = host.model.layers[-2]
    original = layer.mlp
    losses, hits = [], []
    try:
        layer.mlp = graft
        for i in range(0, len(ids), 8):
            batch = ids[i:i+8]
            z = host(mx.array(batch[:, :64]))[:, P].astype(mx.float32)
            y = mx.array(batch[:, POS+1])
            losses.append(np.array(nn.losses.cross_entropy(z, y, reduction="none")).ravel())
            hits.append(np.array(z.argmax(-1) == y).ravel())
    finally:
        layer.mlp = original
    return np.concatenate(losses), np.concatenate(hits)


def ckpt(arm, seed):
    return WORK / f"block-{arm}-{seed}.safetensors"


def fit(host, arm, seed, train, graft, validation_ids, initial):
    block = host.model.layers[-1]
    master = tree_map(lambda x: x.astype(mx.float32), initial)
    block.update(bf16(master))
    opt = optim.Adam(2e-5, bias_correction=True)
    order = mx.array(np.random.default_rng(12000+seed).permutation(len(train[arm])).astype(np.int32))
    assert len(order) == CHECKS[-1] * 16
    index = order.reshape(-1, 16)

    def objective(w, x, target):
        block.update(bf16(w))
        h = scored_output(block, x).astype(mx.float32)
        t = target.astype(mx.float32)
        return ((mx.square(h-t).mean(-1)) / (mx.square(t).mean(-1) + .01)).mean()

    value_grad = mx.value_and_grad(objective)
    # Keep all checkpoint decisions on the original validation split.
    history = []
    best = float("inf")
    started = monotonic()
    for step in range(97):
        loss = None
        if step:
            ix = index[step-1]
            loss, grads = value_grad(master, train[arm][ix], train["teacher"][ix])
            master = opt.apply_gradients(grads, master)
            mx.eval(master, opt.state, loss)
        if step in CHECKS:
            block.update(bf16(master))
            nll, hits = live_scores(host, graft, validation_ids)
            score = float(nll.astype("float64").mean())
            history.append(dict(step=step, nll=score, accuracy=float(hits.mean()),
                                last_batch_objective=float(loss.item()) if loss is not None else None))
            if score < best:
                chosen, best = step, score
                block.save_weights(str(ckpt(arm, seed)))
    block.update(initial)
    return dict(selected=chosen, validation=history, seconds=round(monotonic()-started, 2))


def main():
    host = load_model("host")
    block = host.model.layers[-1]
    initial = tree_map(lambda x: x, block.parameters())
    counts = {k: v.size for k, v in tree_flatten(block.trainable_parameters())}
    train = mx.load(str(WORK / "train.safetensors"))
    validation_ids = tokens("val")
    grafts = modules(host, load_model("donor"), load_model("general"))
    dest = RESULTS / "selection.json"
    selection = json.loads(dest.read_text()) if dest.exists() else {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            if key in selection and ckpt(arm, seed).exists():
                continue
            result = fit(host, arm, seed, train, grafts[arm], validation_ids, initial)
            selection[key] = result
            write(dest, selection)
            print(key, result["selected"], result["validation"][-1]["nll"], result["seconds"], flush=True)
    write(RESULTS / "trainable.json", dict(parameter_count=sum(counts.values()), tensors=counts,
                                           device=mx.device_info()["device_name"]))


if __name__ == "__main__":
    main()
