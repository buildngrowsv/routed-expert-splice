"""Held-out scores, paired donor withdrawal, crossover and live model parity."""
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from features import ARMS, CACHE, P, POS, RESULTS, WORK, load_model, modules, write
from forced import SEEDS
from train import ckpt, live_scores, measure


def main():
    assert len(json.loads((RESULTS / "selection.json").read_text())) == len(ARMS) * len(SEEDS)
    host, coder, general = [load_model(s) for s in ("host", "donor", "general")]
    arms = modules(host, coder, general)
    block = host.model.layers[-1]
    initial = tree_map(lambda x: x, block.parameters())
    table = mx.load(str(WORK / "upstream.safetensors"))
    with np.load(CACHE / "upstream-tokens.npz") as z:
        ids = z["tokens"]
        domain, path = (np.repeat(z[k], len(POS)) for k in ("domain", "path"))
    labels = mx.array(ids[:, POS+1])
    data = {"domain": domain, "paths": path}
    parity, changes = {}, {}
    original = host.model.layers[-2].mlp
    sample = np.array([0, 65, 193, 255, 511, 512, 733, 1023])

    def save(name, a, b):
        data[name+"_nll"] = np.stack(a)
        data[name+"_correct"] = np.stack(b)

    def check_live(arm):
        host.model.layers[-2].mlp = arms[arm]
        try:
            z = host(mx.array(ids[sample, :64]))[:, P].astype(mx.float32)
            y = labels[mx.array(sample)]
            live = np.array(nn.losses.cross_entropy(z, y, reduction="none")).ravel()
            loss, hit = measure(host, block, table[arm][mx.array(sample)], y)
            from features import scored_output
            proxy = host.model.embed_tokens.as_linear(host.model.norm(scored_output(block, table[arm][mx.array(sample)])))
            top = float((np.array(z.argmax(-1)) == np.array(proxy.argmax(-1))).mean())
            error = np.abs(live-loss)
            return dict(tokens=len(live), top1_agreement=top, mean_abs_nll=float(error.mean()),
                        max_abs_nll=float(error.max()))
        finally:
            host.model.layers[-2].mlp = original

    for arm in ARMS:
        block.update(initial)
        nll, correct = live_scores(host, arms[arm], ids)
        save("frozen_"+arm, [nll], [correct])
        ls, hs = [], []
        switches = {"coder": ("knockout", "shuffled"), "shuffled": ("coder",)}.get(arm, ())
        crosses = {s: ([], []) for s in switches}
        for seed in SEEDS:
            block.load_weights(str(ckpt(arm, seed)))
            delta = {}
            for (k, old), (j, new) in zip(tree_flatten(initial), tree_flatten(block.parameters())):
                assert k == j
                delta[k] = float(mx.max(mx.abs(new.astype(mx.float32) - old.astype(mx.float32))).item())
            changes[f"{arm}-{seed}"] = delta
            l, h = live_scores(host, arms[arm], ids)
            ls.append(l)
            hs.append(h)
            for other, (lo, hi) in crosses.items():
                l, h = live_scores(host, arms[other], ids)
                lo.append(l)
                hi.append(h)
            if seed == SEEDS[0]:
                parity[arm] = check_live(arm)
        save("trained_"+arm, ls, hs)
        for other, (ls, hs) in crosses.items():
            save(arm+"_on_"+other, ls, hs)
        print("Held-out evaluations", arm, parity[arm], flush=True)
    write(RESULTS / "full-model-check.json", parity)
    write(RESULTS / "weight-changes.json", changes)
    np.savez_compressed(RESULTS / "tokens.npz", **data)
    # All released scores above come from the ordinary full model. Proxy discrepancies
    # are diagnostics of the training cache and do not enter the reported metrics.


if __name__ == "__main__":
    main()
