"""Recount the complete repairs, matched controls and donor-removal groups."""
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
ARMS = ("native", "knockout", "direct", "shuffled", "general", "coder", "refit-shuffled", "refit")
REMOVALS = ("refit-to-knockout", "refit-to-native", "refit-to-old-input", "refit-to-shuffled")


def read(name):
    return json.loads((ROOT/(name+".json")).read_text())


def main():
    receipt = read("receipt")
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "23d035eb6928d5a3c306a23d8ebf206faee0f8d1"
    result = read("result")
    rows = read("programs")
    train_ids = read("training_ids")
    archived = json.loads((ROOT.parents[1]/"results/fit/training_ids.json").read_text())
    assert train_ids==archived and len(set(train_ids))==16
    meta = read("inputs")
    assert meta["host_block"]==27 and meta["trained_block"]==28 and meta["calibration"]=="ordinary"
    assert len({r["prompt"] for r in rows})==768
    gold = []
    for i,r in enumerate(rows):
        j,k = divmod(i,3)
        assert r["id"]==f"opener-{60000+j}-{k}"
        if k==2:
            assert [v for v in zip(*(a["prompt"] for a in rows[i-2:i+1]),strict=True)
                    if len(set(v))>1]==[("(","[","{")]
        before,after = r["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        parsed = []
        for p,token in enumerate((")","]","}",":")):
            try:
                ast.parse(before+token+after)
            except SyntaxError:
                continue
            parsed.append(p)
        assert parsed==[k] and r["gold"]==k
        gold.append(k)
    gold = np.array(gold)
    for arm in ARMS:
        meta = read(arm+"-complete")
        assert meta["steps"]==len(meta["losses"])==8 and meta["trainable"]==275268608
        assert np.isfinite(meta["losses"]).all()
    grouped, items, choices = {}, {}, {}
    for k in ("standalone-donor",*ARMS,*REMOVALS,*["before-"+k for k in ARMS]):
        x = np.load(ROOT/(k+".npy"))
        assert x.shape==(768,4) and np.isfinite(x).all()
        pred = x.argmax(1)
        items[k] = int((pred==gold).sum())
        choices[k] = np.bincount(pred,minlength=4).tolist()
        grouped[k] = (pred==gold).reshape(256,3).all(1)
    complete = {k:int(v.sum()) for k,v in grouped.items()}
    assert result["individual"]==items and result["complete"]==complete
    refit = grouped["refit"]
    new = refit & (~grouped["native"]) & (~grouped["before-native"])
    strict = new & (~grouped["refit-to-knockout"]) & (~grouped["refit-to-native"])
    def ids(mask):
        return [f"opener-{60000+i}" for i in np.flatnonzero(mask)]
    checks = {"new_ids":new,"strict_ids":strict,
              "strict_overlap_with_standalone_donor":strict & grouped["standalone-donor"] & ~grouped["before-native"],
              "strict_removed_by_old_input":strict & ~grouped["refit-to-old-input"],
              "off_losses":refit & ~grouped["refit-to-knockout"],"off_rescues":~refit & grouped["refit-to-knockout"],
              "native_regressions":~refit & grouped["native"]}
    for key,mask in checks.items():
        assert result[key]==ids(mask)
    bootstrap = np.random.default_rng(688431).choice(len(refit), (5000,len(refit)))
    for k,report in result["advantages"].items():
        delta = refit.astype(int)-grouped[k]
        ci = np.quantile(delta[bootstrap].mean(1),[.025,.975])
        np.testing.assert_allclose([delta.mean(),*ci],[report["mean"],*report["interval"]])
    passes = strict.sum()>=10 and all(v["mean"]>=.05 and v["interval"][0]>0 for v in result["advantages"].values())
    assert result["decision"]==("replicate" if passes else "stop")
    print(json.dumps({**result,"choices":choices},indent=2))


if __name__=="__main__":
    main()
