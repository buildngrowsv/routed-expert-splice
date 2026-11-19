"""CPU-only checks of domain scores, contrasts, checkpoint choices and holdout isolation."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ORIGINAL = HERE.parent / "results"


def read(path):
    return json.loads(path.read_text())


def main():
    orig = read(ORIGINAL / "windows.json")
    test_paths = {p for split, _, p, _, _ in orig if split == "test"}
    train_val = {p for split, _, p, _, _ in orig if split != "test"}
    assert not test_paths & train_val
    prior = [r[1:] for r in orig]
    for study in ("forced", "donors"):
        prior.extend(read(ORIGINAL / study / "windows.json"))
    old_locations = {(p, off) for _, p, off, _ in prior}
    old_hashes = {h for _, _, _, h in prior}
    current = read(R / "windows.json")
    assert len(current) == 1024
    for _, p, off, h in current:
        assert p in test_paths and (p, off) not in old_locations and h not in old_hashes
        assert off % 65 == 0
        old_locations.add((p, off))
        old_hashes.add(h)

    report = read(R / "summary.json")
    selection = read(R / "selection.json")
    changes = read(R / "weight-changes.json")
    assert len(selection) == 18
    for key, run in selection.items():
        assert [x["step"] for x in run["validation"]] == [0, 8, 32, 96]
        assert run["selected"] == min(run["validation"], key=lambda x:x["nll"])["step"]
        assert bool(max(changes[key].values())) == bool(run["selected"])

    with np.load(R / "tokens.npz") as data:
        np.testing.assert_array_equal(data["paths"], np.repeat([r[1] for r in current], 4))
        np.testing.assert_array_equal(data["domain"], np.repeat([r[0] for r in current], 4))
        for i, d in enumerate(("code", "docs")):
            ix = data["domain"] == i
            assert ix.sum() == 2048
            for k, m in report["metrics"][d].items():
                losses, hits = data[k+"_nll"][:,ix], data[k+"_correct"][:,ix]
                assert np.isfinite(losses).all() and losses.shape == hits.shape
                np.testing.assert_allclose([losses.mean(dtype=float),hits.mean()], [m["nll"],m["accuracy"]])
            def scores(k):
                return data[k+"_nll"][:,ix].astype(float)
            interaction = ((scores("coder_on_knockout")-scores("trained_coder"))
                           -(scores("frozen_knockout")-scores("frozen_coder")))
            np.testing.assert_allclose(interaction.mean(axis=1), report["comparisons"][d]["increased_use"]["seed_means"])
            swapped = (scores("coder_on_shuffled")+scores("shuffled_on_coder")
                       -scores("trained_coder")-scores("trained_shuffled"))/2
            np.testing.assert_allclose(swapped.mean(), report["comparisons"][d]["cross_match"]["mean"])
    print("Verified splits, 18 checkpoint choices, weight updates, all domain scores and the two use diagnostics.")


if __name__ == "__main__":
    main()
