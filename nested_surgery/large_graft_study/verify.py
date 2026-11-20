"""Rebuild the item ledger, intervention overlaps, and paired contrasts on fresh repairs."""
import ast
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ("native", "knockout", "direct", "shuffled", "general", "coder")


def read(name):
    return json.loads((RESULTS / name).read_text())


def check_prompts(rows):
    old = json.loads((HERE.parent / "opener_study/results/programs.json").read_text())
    assert {r["prompt"] for r in rows}.isdisjoint(r["prompt"] for r in old)
    assert len(rows) == len(set(r["prompt"] for r in rows)) == 768
    labels = []
    for offset in range(0, len(rows), 3):
        batch = rows[offset:offset + 3]
        base = 30000 + offset // 3
        assert [r["id"] for r in batch] == [f"opener-{base}-{k}" for k in range(3)]
        prompts = [r["prompt"] for r in batch]
        assert len(set(map(len, prompts))) == 1
        assert [p for p in zip(*prompts) if len(set(p)) > 1] == [("(", "[", "{")]
        for row in batch:
            prefix, suffix = row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
            assert suffix == "\n    return x\n"
            correct = []
            for k, patch in enumerate((")", "]", "}", ":")):
                try:
                    ast.parse(prefix + patch + suffix)
                except SyntaxError:
                    continue
                correct.append(k)
            assert correct == [row["gold"]]
            labels.extend(correct)
    return np.array(labels)


def main():
    rows = read("programs.json")
    gold = check_prompts(rows)
    stated = read("result.json")
    archived_ids = json.loads((HERE.parent / "results/fit/training_ids.json").read_text())
    assert read("training_ids.json") == archived_ids and len(set(archived_ids)) == 16
    decisions = {}
    answers = {}
    intact = {}
    for name in [*ARMS, *["before-" + x for x in ARMS], *["coder-to-" + x for x in ("knockout", "native", "shuffled")]]:
        logits = np.load(RESULTS / (name + ".npy"))
        assert logits.shape == (len(rows), 4) and np.isfinite(logits).all()
        answers[name] = logits.argmax(1)
        correct = answers[name] == gold
        intact[name] = correct.reshape(-1, 3).all(1)
        decisions[name] = {"prompts": int(correct.sum()), "groups": int(intact[name].sum())}
        assert decisions[name] == dict(prompts=stated["individual"][name], groups=stated["complete"][name])
    for arm in ARMS:
        meta = read(arm + "-complete.json")
        assert meta["trainable"] == 275268608 and meta["steps"] == len(meta["losses"]) == 8
        assert np.isfinite(meta["losses"]).all()
    def ids(mask):
        return [rows[3 * i]["id"].rsplit("-", 1)[0] for i in np.flatnonzero(mask)]
    coder = intact["coder"]
    knocked = intact["coder-to-knockout"]
    restored = intact["coder-to-native"]
    new = coder & ~intact["native"] & ~intact["before-native"]
    strict = new & ~knocked & ~restored
    for label, mask in (("new_ids", new), ("strict_ids", strict), ("off_losses", coder & ~knocked),
                        ("off_rescues", knocked & ~coder), ("native_regressions", intact["native"] & ~coder)):
        assert ids(mask) == stated[label]
    index = np.random.default_rng(688431).choice(len(coder), size=(5000, len(coder)))
    for arm, claim in stated["advantages"].items():
        delta = coder.astype(int) - intact[arm].astype(int)
        lo, hi = np.quantile(delta[index].mean(1), [.025, .975])
        np.testing.assert_allclose([delta.mean(), lo, hi], [claim["gap"], *claim["interval"]])
    gated = len(ids(strict)) >= 10 and all(
        stated["advantages"][x]["gap"] >= .05 and stated["advantages"][x]["interval"][0] > 0
        for x in ("native", "shuffled"))
    assert ("replicate" if gated else "stop") == stated["decision"]
    companion = read("standalone-result.json")
    with np.load(RESULTS / "standalone.npz") as z:
        np.testing.assert_array_equal(z["gold"], gold)
        np.testing.assert_array_equal(z["host"].argmax(1), answers["before-native"])
        donor_only = ((z["donor"].argmax(1) == gold).reshape(-1, 3).all(1) & ~intact["before-native"])
    assert sorted(ids(donor_only)) == companion["standalone_donor_only"]
    assert sorted(ids(strict & donor_only)) == companion["strict_graft_new_on_donor_only"]
    assert sorted(ids(strict & ~donor_only)) == companion["strict_graft_new_elsewhere"]
    print(json.dumps(dict(group_count=len(coder), scores=decisions, standalone_donor_only=len(ids(donor_only)),
                          new_groups=int(new.sum()), strict_groups=int(strict.sum()),
                          strict_standalone_overlap=int((strict & donor_only).sum()),
                          off_losses=int((coder & ~knocked).sum()), off_rescues=int((knocked & ~coder).sum()),
                          native_regressions=int((intact["native"] & ~coder).sum()),
                          advances=bool(gated)), indent=2))


if __name__ == "__main__":
    main()
