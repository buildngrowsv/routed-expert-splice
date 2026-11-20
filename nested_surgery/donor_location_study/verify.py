"""Audit the source substitution, prompt outcomes and decision from saved logits."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
PANEL = HERE.parent / "large_graft_study/results"


def check():
    result = json.loads((ROOT / "result.json").read_text())
    rows = json.loads((PANEL / "programs.json").read_text())
    gold = np.asarray([r["gold"] for r in rows])
    groups = np.arange(len(rows)).reshape(-1, 3)
    label = [r["id"].rsplit("-", 1)[0] for r in rows[::3]]
    assert len(set(label)) == 256
    triplets, answers, margins, ties = {}, {}, {}, {}
    with np.load(ROOT / "interventions.npz") as z:
        np.testing.assert_array_equal(gold, z["gold"])
        changed, same, lengths = z["changed_source_ids"], z["same_source_ids"], z["lengths"]
        np.testing.assert_array_equal(changed, groups[:, [1, 2, 0]].ravel())
        assert np.all(gold[same] == gold) and np.all(gold[changed] != gold)
        for i, picked in enumerate(same):
            eligible = [j for j in range(len(rows)) if gold[j] == gold[i] and j // 3 != i // 3]
            selected = min(eligible, key=lambda j: (abs(int(lengths[j]) - int(lengths[i])), (j - i) % len(rows)))
            assert selected == picked
        exact = int(np.count_nonzero(lengths == lengths[same]))
        assert exact == result["exact_length_shams"]
        for k in ("intact", "off", "same", "changed"):
            x = z[k]
            assert x.shape == (768, 4) and np.isfinite(x).all()
            margins[k] = x[np.arange(len(x)), gold[changed]] - x[np.arange(len(x)), gold]
            ties[k] = int(np.count_nonzero((x == x.max(axis=1, keepdims=True)).sum(1) > 1))
            answers[k] = np.argmax(x, axis=1)
            correct = answers[k] == gold
            triplets[k] = correct[groups].all(1)
            assert int(correct.sum()) == result["item_scores"][k]
            assert int(triplets[k].sum()) == result["group_scores"][k]
    with np.load(PANEL / "standalone.npz") as standalone:
        np.testing.assert_array_equal(standalone["donor"].argmax(1), answers["intact"])
    for k, solved in triplets.items():
        assert int((triplets["intact"] & ~solved).sum()) == result["intact_losses"][k]
        assert int((~triplets["intact"] & solved).sum()) == result["intact_rescues"][k]
    source_correct = answers["intact"] == gold
    broken = source_correct & (answers["changed"] != gold)
    switched = broken & (answers["changed"] == gold[changed])
    assert int(switched.sum()) == result["switched_to_new_opener"]
    assert int(source_correct.sum()) == result["originally_correct"]
    target = triplets["intact"] & triplets["same"] & ~triplets["changed"]
    np.testing.assert_array_equal(np.flatnonzero(target), result["targeted_ids"])
    d = triplets["same"].astype(int) - triplets["changed"].astype(int)
    ix = np.random.default_rng(721051).choice(len(d), size=(5000, len(d)))
    ci = np.quantile(d[ix].mean(1), [.025, .975])
    np.testing.assert_allclose([d.mean(), *ci], [result["same_minus_changed"], *result["interval"]])
    passes = d.mean() >= .05 and ci[0] > 0 and target.sum() >= 10
    assert result["decision"] == ("advance" if passes else "stop")
    movement = (margins["changed"] - margins["same"]).astype(float).reshape(-1, 3).mean(1)
    margin_ci = np.quantile(movement[ix].mean(1), [.025, .975])
    return dict(changed_minus_same_source_target_log_odds=movement.mean(), margin_interval=margin_ci.tolist(),
                ties_at_maximum=ties, same_preferred_donor_target=int((answers["same"] == gold[changed]).sum()),
                changed_preferred_donor_target=int((answers["changed"] == gold[changed]).sum()),
                group_scores=result["group_scores"], item_scores=result["item_scores"],
                targeted_groups=[k for k, hit in zip(label, target) if hit], unique_same_sources=len(set(same.tolist())),
                exact_length_matches=exact, broken_originally_correct=int(broken.sum()), switched_to_new_opener=int(switched.sum()),
                per_opener_switched=[int((switched & (gold == k)).sum()) for k in range(3)],
                intact_lost_to_zero=result["intact_losses"]["off"], newly_correct_with_zero=result["intact_rescues"]["off"],
                delta=d.mean(), interval=ci.tolist(), passed=bool(passes))


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
