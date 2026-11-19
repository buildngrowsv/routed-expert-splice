"""Build full-context entrances to the final host block for matched interventions."""
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_donors import GENERAL
from forced import Replacement, new_test, states
from prepare import CACHE, OUT, POS, create_attention_mask, load_model, ridge, write

WORK = CACHE / "penultimate"
RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("native", "knockout", "direct", "shuffled", "general", "coder")
SPLITS = ("train", "val", "upstream")
P = mx.array(POS)


class Zero(nn.Module):
    def __call__(self, x):
        return mx.zeros_like(x)


class Affine(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w, self.b = w, b

    def __call__(self, u):
        return (u.astype(mx.float32) @ self.w + self.b).astype(mx.bfloat16)


def residual_and_input(host, ids):
    core = host.model
    x = core.embed_tokens(ids)
    mask = create_attention_mask(x, None)
    for layer in core.layers[:-2]:
        x = layer(x, mask)
    layer = core.layers[-2]
    h = x + layer.self_attn(layer.input_layernorm(x), mask)
    return h, layer.post_attention_layernorm(h)


def scored_output(block, x):
    # Attention sees the full sequence. The final per-token MLP has no cross-token edges.
    mask = create_attention_mask(x, None)
    h = (x + block.self_attn(block.input_layernorm(x), mask))[:, P]
    return h + block.mlp(block.post_attention_layernorm(h))


def tokens(split):
    with np.load(CACHE / f"{split}-tokens.npz") as data:
        return data["tokens"]


def training_rows(host, general, ids):
    arrays = [[], [], []]
    for i in range(0, len(ids), 16):
        x = mx.array(ids[i:i+16, :64])
        _, u = residual_and_input(host, x)
        v = host.model.layers[-2].mlp(u[:, P])
        g = states(general, x)[1][:, P]
        for acc, arr in zip(arrays, (u[:, P], v, g)):
            acc.append(np.array(arr.astype(mx.float32)).reshape(-1, arr.shape[-1]))
    return [np.concatenate(a) for a in arrays]


def modules(host, coder, general, fit=False):
    orig = host.model.layers[-2].mlp
    arms = {"native": orig, "knockout": Zero()}
    path = WORK / "interfaces.safetensors"
    if fit:
        u, v, gen_input = training_rows(host, general, tokens("train"))
        with np.load(CACHE / "donor-train.npz") as data:
            coder_input = data["u"]
        weights = {}
        for name, source in (("coder", coder_input), ("general", gen_input), ("direct", v)):
            w, b = ridge(u, source)
            weights[name + "_w"], weights[name + "_b"] = mx.array(w), mx.array(b)
    else:
        weights = mx.load(str(path))
    arms["direct"] = Affine(weights["direct_w"], weights["direct_b"])
    for arm in ("coder", "shuffled", "general"):
        source = "general" if arm == "general" else "coder"
        small = general if arm == "general" else coder
        graft = Replacement(small.model.layers[-1].mlp, "shuffled" if arm == "shuffled" else "intact",
                            weights[source + "_w"], weights[source + "_b"], None, None, Zero())
        if fit:
            rows = []
            for i in range(0, len(u), 128):
                rows.append(np.array(graft.expert_output(mx.array(u[i:i+128])).astype(mx.float32)))
            wo, bo = ridge(np.concatenate(rows), v)
            weights[arm + "_out"], weights[arm + "_bias"] = mx.array(wo), mx.array(bo)
        graft.wout, graft.bout = weights[arm + "_out"], weights[arm + "_bias"]
        arms[arm] = graft
    if fit:
        mx.save_safetensors(str(path), weights)
    return arms


def run():
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    prior = sum((json.loads((OUT / part / "windows.json").read_text()) for part in ("forced", "donors")), [])
    write(WORK / "prior.json", prior)
    new_test(WORK / "prior.json", 158891, "upstream", RESULTS / "windows.json")
    snapshot_download("Qwen/Qwen2.5-0.5B", revision=GENERAL, local_dir=CACHE/"general",
                      allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"])
    host, coder, general = [load_model(n) for n in ("host", "donor", "general")]
    arms = modules(host, coder, general, fit=True)
    print("Fitted all training-only interfaces", flush=True)
    for split in SPLITS:
        ids = tokens(split)
        chunks = {k: [] for k in (*ARMS, "teacher")}
        for i in range(0, len(ids), 16):
            h, u = residual_and_input(host, mx.array(ids[i:i+16, :64]))
            batch = {k: h + graft(u) for k, graft in arms.items()}
            batch["teacher"] = scored_output(host.model.layers[-1], batch["native"])
            mx.eval(batch)
            for k, x in batch.items():
                chunks[k].append(x)
        mx.save_safetensors(str(WORK / f"{split}.safetensors"), {k: mx.concatenate(v) for k, v in chunks.items()})
        print("Cached", split, len(ids), flush=True)


if __name__ == "__main__":
    run()
