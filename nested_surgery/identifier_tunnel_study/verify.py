"""Recover retained pairs, double reversals and intervals from saved logit margins."""
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
MODES = ("native","learned_clean","learned_swap","random_clean","random_swap")


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "2a25de6f6e74cc491e881d2e6f8cfa640d5e9f9a"
    result = json.loads((ROOT/"result.json").read_text())
    rows = json.loads((ROOT/"test-programs.json").read_text())
    assert result["reference_blocks"] == 7 and result["groups"] == 512
    assert [r["id"] for r in rows] == list(range(12000,12512))
    for row in rows:
        assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b] == [row["site"]]
        for p,letter in zip(row["prompts"],row["letters"],strict=True):
            env = {}
            exec(compile(ast.parse(p+letter+"\n"),"<archived generated identifier>","exec"),env)
            assert env["fetch"]() == 7
    audit = {}
    for model in ("donor","host"):
        with np.load(ROOT/(model+"-margins.npz")) as data:
            assert set(data.files) == set(MODES)
            scores = {k:data[k] for k in MODES}
        assert all(a.shape == (512,2) and np.isfinite(a).all() for a in scores.values())
        clean = (scores["native"]>0).all(1)
        valid_learned = clean & (scores["learned_clean"]>0).all(1)
        valid_random = clean & (scores["random_clean"]>0).all(1)
        population = valid_learned & valid_random
        flipped_learned = (scores["learned_swap"]<0).all(1)
        flipped_random = (scores["random_swap"]<0).all(1)
        n, l, n_random, common = (int(x.sum()) for x in (clean,valid_learned,valid_random,population))
        flips = int((flipped_learned & valid_learned).sum())
        a,b = (int(flags[population].sum()) for flags in (flipped_learned,flipped_random))
        diff = flipped_learned[population].astype(int)-flipped_random[population]
        bounds = [0.,0.]
        if common:
            samples = np.random.default_rng(60217).integers(common,size=(4000,common))
            bounds = np.quantile(diff[samples].mean(1),[.025,.975]).tolist()
        delta = float(diff.mean()) if common else 0.
        qualify = n>=100 and l>=.9*n and flips>=.8*l and common>=100 and delta>=.2 and bounds[0]>0
        summary = dict(native=n,learned_retained=l,random_retained=n_random,matched_eligible=common,
                       learned_reversed=flips,learned_reversed_on_matched=a,random_reversed_on_matched=b,
                       contrast=dict(mean=delta,interval=bounds),qualifies=qualify)
        assert summary == result["arms"][model]
        audit[model] = dict(**summary, mean_margin={k:float(v.mean()) for k,v in scores.items()})
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
