"""Calculate registered paired contrasts from released per-position scores."""
import numpy as np
from features import ARMS, RESULTS, write
from run import bootstrap_files


def main():
    with np.load(RESULTS / "tokens.npz") as z:
        domain, paths = z["domain"], z["paths"]
        losses = {k[:-4]: z[k].astype("float64") for k in z.files if k.endswith("_nll")}
        hits = {k: z[k+"_correct"] for k in losses}
    metrics, comparisons = {}, {}
    gain = {a: losses["frozen_"+a] - losses["trained_"+a] for a in ARMS}
    off_before = losses["frozen_knockout"] - losses["frozen_coder"]
    off_after = losses["coder_on_knockout"] - losses["trained_coder"]
    differences = {
        **{"gain_"+a: g for a,g in gain.items()},
        "off_before": off_before, "off_after": off_after, "increased_use": off_after-off_before,
        "cross_match": .5 * (losses["coder_on_shuffled"] + losses["shuffled_on_coder"]
                             - losses["trained_coder"] - losses["trained_shuffled"]),
        **{"gain_coder_minus_"+a: gain["coder"]-g for a,g in gain.items() if a != "coder"},
        **{"trained_"+a+"_minus_coder": losses["trained_"+a]-losses["trained_coder"]
           for a in ARMS if a != "coder"},
        "top1_coder_gain": hits["trained_coder"].astype(float)-hits["frozen_coder"],
    }
    for d, name in enumerate(("code", "docs")):
        ix = domain == d
        metrics[name] = {k: dict(nll=float(v[:, ix].mean()), accuracy=float(hits[k][:, ix].mean()))
                         for k, v in losses.items()}
        comparisons[name] = {}
        for k, v in differences.items():
            diff = v[:, ix]
            comparisons[name][k] = dict(mean=float(diff.mean()), interval=bootstrap_files(diff, paths[ix]),
                                        seed_means=diff.mean(1).tolist())
    write(RESULTS / "summary.json", dict(metrics=metrics, comparisons=comparisons,
                                        file_counts={d:int(len(np.unique(paths[domain==i])))
                                                     for i,d in enumerate(("code", "docs"))}))


if __name__ == "__main__":
    main()
