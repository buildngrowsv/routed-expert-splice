"""Audit every candidate repair and reconstruct the paired format comparisons."""
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
FORMATS = ("fim", "empty", "prefix")
MODELS = ("donor", "host")
PATCHES = (")", "]", "}", ":")


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "ab1dd787cabdb8d1bd1c341df6010d175ac2da7d"
    summary = json.loads((ROOT/"result.json").read_text())
    rows = json.loads((ROOT/"programs.json").read_text())
    assert [row["id"] for row in rows] == [f"opener-{i}-{k}" for i in range(80000,80256) for k in range(3)]
    gold = np.asarray([row["gold"] for row in rows])
    for start in range(0,len(rows),3):
        group = rows[start:start+3]
        assert [row["gold"] for row in group] == [0,1,2]
        assert all(sum(x!=y for x,y in zip(group[0]["prompt"],row["prompt"],strict=True))==1 for row in group[1:])
        for row in group:
            p,s = row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
            answers = []
            for candidate, patch in enumerate(PATCHES):
                try:
                    parsed = ast.parse(p+patch+s)
                except SyntaxError:
                    continue
                namespace = {"__builtins__":{"sum":sum}}
                exec(compile(parsed,"<archived generated opener>","exec"),namespace)
                assert [namespace["transform"](x) for x in (-19,0,11)] == [-19,0,11]
                answers.append(candidate)
            assert answers == [row["gold"]]
    hits, counts = {}, {}
    with np.load(ROOT/"format-scores.npz") as data:
        assert set(data.files) == {"gold",*(f"{model}_{fmt}" for model in MODELS for fmt in FORMATS)}
        np.testing.assert_array_equal(gold,data["gold"])
        for name in data.files:
            if name == "gold":
                continue
            values = data[name]
            assert values.shape == (768,4) and np.isfinite(values).all()
            picks = values.argmax(1)
            hits[name] = (picks==gold).reshape(256,3).all(1)
            counts[name] = np.bincount(picks,minlength=4).tolist()
    correct = {k:int(v.sum()) for k,v in hits.items()}
    assert correct == summary["complete"] and counts == summary["candidate_counts"]
    deltas = {f:hits[f"donor_{f}"].astype(int)-hits[f"host_{f}"] for f in FORMATS}
    samples = np.random.default_rng(245822).integers(256,size=(5000,256))
    def measure(x):
        return dict(mean=float(x.mean()),interval=np.percentile(x[samples].mean(1),[2.5,97.5]).tolist())
    result = {fmt:measure(x) for fmt,x in deltas.items()}
    assert result == summary["gaps"]
    assert measure(deltas["prefix"]-deltas["fim"]) == summary["prefix_minus_fim_gap"]
    prefix = result["prefix"]
    passed = prefix["mean"]>=.10 and prefix["interval"][0]>0
    assert summary["decision"] == ("portable_advantage" if passed else "format_specific_or_unresolved")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
