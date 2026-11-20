"""Audit contrast-only cross-width alignment and full-vocabulary outcomes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOS = HERE.parent
RECIPIENTS = {
    "32b": ("qwen-cross-width-method-control-32b", "9cef06213271ed3e72e8c089ae89af82f917238d", 5120),
    "72b": ("qwen-cross-width-72b", "d2609dcba899b8755b47a8862512aca8e8401326", 8192),
}


def verify_geometry(path, width):
    with np.load(path) as data:
        x, y = data["x"], data["y"]
        assert x.shape == (256, 7) and y.shape == (256, width)
        assert all(np.isfinite(data[k]).all() for k in data.files)
        gram = x.T @ x
        penalty = np.diag(gram).mean() * .02
        np.testing.assert_allclose(data["penalty"], penalty)
        expected = np.linalg.solve(gram + penalty * np.eye(7), x.T @ y)
        np.testing.assert_allclose(data["write"], expected.T, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(data["read"].T @ data["write"], np.eye(7), atol=1e-8)
        np.testing.assert_allclose(data["random"].T @ data["write"], np.zeros((7, 7)), atol=1e-8)
        np.testing.assert_allclose(data["random"].T @ data["random"], np.eye(7), atol=1e-8)
        assert data["center_input"].shape == (width,) and data["center_target"].shape == (7,)
        return dict(penalty=float(penalty), in_sample_squared_residual_ratio=float(
            np.square(y - x @ expected).sum() / np.square(y).sum()))


def verify_data(recipient):
    job, revision, width = RECIPIENTS[recipient]
    root = HERE / recipient
    receipt = json.loads((root / "receipt.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (job, revision, "done", 0)
    result = json.loads((root / "result.json").read_text())
    panel_archive = REPOS / "numeric_scale_study/results/programs.json"
    map_archive = REPOS / "numeric_absolute_study/results/map.npz"
    assert sha256(panel_archive.read_bytes()).hexdigest() == result["prompts_digest"]
    assert sha256(map_archive.read_bytes()).hexdigest() == result["map_digest"]
    calibration = json.loads((root / "calibration.json").read_text())
    assert [row["id"] for row in calibration] == list(range(125000, 125256))
    assert len(json.loads(panel_archive.read_text())["host"]) == 512
    numerical = verify_geometry(root / "alignment.npz", width)
    scalar = REPOS / "numeric_scalar_study/results/scalar-32b.npz"
    with np.load(root / (recipient + ".npz")) as current, np.load(scalar) as previous:
        names = ("stitch", "random", "whole")
        for key in (*names, "native"):
            assert current[key].shape == (512, 2, 4) and np.isfinite(current[key]).all()
        norms = current["norms"]
        assert norms.shape == (1024, 2) and np.isfinite(norms).all() and (norms >= 0).all()
        np.testing.assert_allclose(norms[:, 0], norms[:, 1], rtol=1e-5, atol=1e-6)
        native = (current["native"][:, :, 0] == 1).all(axis=1)
        old_native = (previous["native"][:, :, 0] == 1).all(axis=1)
        shared = native & old_native
        hits = {m: (current[m][:, :, 1] == 1).all(axis=1) for m in names}
        hits["scalar_32b"] = (previous["borrowed_raw"][:, :, 1] == 1).all(axis=1)
    n = int(native.sum())
    assert result["eligible"] == n and result["shared_eligible"] == int(shared.sum())
    assert result["reversals"] == {m: int(hits[m][native].sum()) for m in names}
    assert result["shared_reversals"] == {m: int(flag[shared].sum()) for m, flag in hits.items()}
    d = hits["stitch"][native].astype(int) - hits["random"][native]
    if n:
        sample = np.random.default_rng(491205).integers(n, size=(5000, n))
        ci = np.percentile(d[sample].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose([d.mean(), *ci], [result["stitch_minus_random"]["mean"],
                                                     *result["stitch_minus_random"]["interval"]])
    passed = (n >= 100 and result["reversals"]["stitch"] >= .75*n and result["reversals"]["whole"] >= .8*n
              and d.mean() >= .20 and ci[0] > 0)
    decision = "alignment_control_transfers" if recipient == "32b" else "cross_width_transfer"
    assert result["decision"] == (decision if passed else "unresolved")
    return dict(**result, alignment_audit=numerical,
                edit_norm_quantiles=np.quantile(norms[:, 0], [.1, .5, .9]).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recipient", choices=RECIPIENTS)
    args = parser.parse_args()
    print(json.dumps(verify_data(args.recipient), indent=2))
