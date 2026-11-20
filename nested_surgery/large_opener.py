"""Qualify a frozen coding expert against a larger frozen general host on identical repairs."""
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
PANEL = HERE / "opener_study" / "results"
CACHE = Path.home() / "research" / "routed-expert-splice" / "cache" / "large-opener"
MODELS = {
    "donor": ("Qwen/Qwen2.5-Coder-7B", "0396a76181e127dfc13e5c5ec48a8cee09938b02"),
    "host": ("Qwen/Qwen2.5-14B", "97e1e76335b7017d8f67c08a19d103c0504298c9"),
}
PATCHES = (")", "]", "}", ":")


def model_files(name):
    repo, revision = MODELS[name]
    dest = CACHE / revision
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / (revision + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = dest / ".downloaded"
        if not marker.is_file():
            snapshot_download(repo, revision=revision, local_dir=dest, max_workers=3,
                              allow_patterns=["config.json", "model*.safetensors", "model.safetensors.index.json", "tokenizer.json"])
            index = json.loads((dest / "model.safetensors.index.json").read_text())
            assert all((dest / f).is_file() for f in set(index["weight_map"].values()))
            marker.touch()
    return dest


def score(path, rows):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    choices = [encode(patch) for patch in PATCHES]
    assert all(len(choice) == 1 for choice in choices)
    assert all(len(encode(tag)) == 1 for tag in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"))
    model, _ = load_model(path, strict=True)
    parts = []
    for start in range(0, len(rows), 8):
        prompts = [r["prompt"] for r in rows[start:start+8]]
        seq = [encode(s) for s in prompts]
        for s, ids in zip(prompts, seq):
            for patch, suffix in zip(PATCHES, choices):
                assert tokenizer.decode(ids + suffix, skip_special_tokens=False) == s + patch
        lengths = np.array([len(ids) for ids in seq])
        padded = np.zeros((len(seq), (max(lengths)+7)//8*8), dtype=np.int32)
        for j, ids in enumerate(seq):
            padded[j, :len(ids)] = ids
        hidden = model.model(mx.array(padded))[mx.arange(len(seq)), mx.array(lengths-1)]
        # Both official architectures use either a separate linear head or the tied embedding matrix.
        projection = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
        parts.append(np.array(projection(hidden)[:, mx.array(choices).flatten()].astype(mx.float32)))
    return np.concatenate(parts)


def main():
    if sys.argv[1:] == ["prefetch"]:
        for name in MODELS:
            print(model_files(name), flush=True)
        return
    rows = json.loads((PANEL / "programs.json").read_text())
    if sys.argv[1:] == ["smoke"]:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        p = Path.home() / "research/routed-expert-splice/cache/host"
        scored = score(p, rows[:8])
        original = np.load(PANEL / "logits.npz")["pre-native"][:8]
        np.testing.assert_array_equal(scored.argmax(1), original.argmax(1))
        print("Archived eight small-host answers reproduced")
        return
    out = Path(os.environ["STUDIO_JOB_DIR"])
    gold = np.array([r["gold"] for r in rows])
    assert np.array_equal(gold, np.tile(np.arange(3), 256))
    results = {}
    for name in MODELS:
        saved = out / (name + ".npy")
        if not saved.exists():
            values = score(model_files(name), rows)
            assert values.shape == (len(gold), 4) and np.isfinite(values).all()
            tmp = out / (name + ".partial.npy")
            np.save(tmp, values)
            os.replace(tmp, saved)
            gc.collect()
        results[name] = np.load(saved).argmax(1) == gold
    donor, host = [results[k].reshape(-1, 3).all(1) for k in ("donor", "host")]
    delta = donor.astype(float) - host
    rng = np.random.default_rng(993613)
    interval = np.quantile(delta[rng.integers(256, size=(5000, 256))].mean(1), [.025, .975])
    offers = np.flatnonzero(donor & ~host)
    report = dict(decision="advance" if len(offers)>=25 and delta.mean()>=.10 and interval[0]>0 else "stop",
        revisions=MODELS, individual={k:int(v.sum()) for k,v in results.items()}, triplets=256,
        complete={"donor":int(donor.sum()), "host":int(host.sum())}, donor_only=offers.tolist(),
        host_only=np.flatnonzero(host & ~donor).tolist(), difference=float(delta.mean()), interval=interval.tolist())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
