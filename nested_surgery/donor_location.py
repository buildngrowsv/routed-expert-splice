"""Exchange only the final MLP contribution in a pretrained coding donor."""
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from large_graft import encode, inputs
from large_opener import model_files
from opener_programs import programs


def matched_sources(lengths, gold):
    """Choose a different group with the same label, nearest length, and stable tie breaking."""
    idx = np.arange(len(gold))
    sources = []
    for i in idx:
        candidates = idx[(gold == gold[i]) & (idx // 3 != i // 3)]
        distance = abs(lengths[candidates] - lengths[i])
        nearest = candidates[distance == distance.min()]
        order = (nearest - i) % len(idx)
        sources.append(nearest[order.argmin()])
    return np.asarray(sources)


def run():
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    out = Path(os.environ["STUDIO_JOB_DIR"])
    rows = programs(256, start=30000)
    assert rows == json.loads((root / "large-qwen-block27-graft/programs.json").read_text())
    gold = np.array([r["gold"] for r in rows])
    np.testing.assert_array_equal(gold, np.tile(np.arange(3), 256))
    path = model_files("donor")
    model, _ = load_model(path, strict=True)
    final = len(model.model.layers) - 1
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    residuals, contributions, lengths = [], [], []
    for i in range(0, len(rows), 8):
        ids, pos, choices, _ = encode(rows[i:i + 8], tokenizer)
        residual, normed = inputs(model, mx.array(ids), final)
        contribution = model.model.layers[final].mlp(normed)
        last = (mx.arange(len(ids)), pos)
        residuals.append(residual[last])
        contributions.append(contribution[last])
        mx.eval(residuals[-1], contributions[-1])
        lengths.extend(np.array(pos).tolist())
    residual = mx.concatenate(residuals)
    update = mx.concatenate(contributions)
    lengths = np.asarray(lengths) + 1
    same = matched_sources(lengths, gold)
    other = np.roll(np.arange(len(rows)).reshape(-1, 3), -1, axis=1).ravel()
    target = gold[other]
    assert np.all(target != gold) and np.all(gold[same] == gold)
    projection = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    logits, complete, answers = {}, {}, {}
    for key, added in (("intact", update), ("off", mx.zeros_like(update)),
                       ("same", update[mx.array(same)]), ("changed", update[mx.array(other)])):
        hidden = model.model.norm(residual + added)
        chunks = []
        for i in range(0, len(rows), 8):
            chunks.append(np.array(projection(hidden[i:i + 8])[:, choices].astype(mx.float32)))
        logits[key] = np.concatenate(chunks)
        assert np.isfinite(logits[key]).all()
        answers[key] = logits[key].argmax(1)
        complete[key] = (answers[key] == gold).reshape(-1, 3).all(1)
    with np.load(root / "large-qwen-fresh-panel-reference/standalone.npz") as z:
        np.testing.assert_array_equal(answers["intact"], z["donor"].argmax(1))
        np.testing.assert_array_equal(gold, z["gold"])
    diff = complete["same"].astype(int) - complete["changed"].astype(int)
    resample = np.random.default_rng(721051).integers(len(diff), size=(5000, len(diff)))
    ci = np.quantile(diff[resample].mean(1), [.025, .975]).tolist()
    targeted = complete["intact"] & complete["same"] & ~complete["changed"]
    np.savez_compressed(out / "interventions.npz", **logits, changed_source_ids=other, same_source_ids=same, lengths=lengths, gold=gold)
    report = dict(
        decision="advance" if diff.mean() >= .05 and ci[0] > 0 and targeted.sum() >= 10 else "stop",
        group_scores={k: int(v.sum()) for k, v in complete.items()},
        item_scores={k: int((v == gold).sum()) for k, v in answers.items()},
        intact_losses={k: int((complete["intact"] & ~v).sum()) for k, v in complete.items()},
        intact_rescues={k: int((~complete["intact"] & v).sum()) for k, v in complete.items()},
        exact_length_shams=int((lengths[same] == lengths).sum()),
        switched_to_new_opener=int(((answers["intact"] == gold) & (answers["changed"] == target)).sum()),
        originally_correct=int((answers["intact"] == gold).sum()),
        same_minus_changed=float(diff.mean()), interval=ci, targeted_ids=np.flatnonzero(targeted).tolist())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
