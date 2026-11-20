"""Recover all repair labels and paired qualification counts from the published prompts and logits."""
import ast
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PANELS = HERE.parent / "opener_study/results"


def main():
    rows = json.loads((PANELS/"programs.json").read_text())
    gold, suffixes = [], set()
    for row in rows:
        prefix, suffix = row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        suffixes.add(suffix)
        valid = []
        for patch in (")","]","}",":"):
            try:
                tree = ast.parse(prefix+patch+suffix)
            except SyntaxError:
                valid.append(False)
                continue
            scope = {"__builtins__":{"sum":sum}}
            exec(compile(tree,"<archived generated repair>","exec"),scope)
            valid.append(all(scope["transform"](x)==x for x in (-173,0,81)))
        assert sum(valid)==1
        gold.append(valid.index(True))
    assert len(suffixes)==1 and len(set(r["prompt"] for r in rows))==len(rows)==768
    np.testing.assert_array_equal(gold,[r["gold"] for r in rows])
    gold=np.array(gold)
    np.testing.assert_array_equal(gold,np.tile(np.arange(3),256))
    for i in range(0,len(rows),3):
        a,b,c=[r["prompt"] for r in rows[i:i+3]]
        assert len(a)==len(b)==len(c)
        assert [(x,y,z) for x,y,z in zip(a,b,c) if len({x,y,z})>1]==[("(","[","{")]
    recorded=json.loads((HERE/"results/result.json").read_text())
    triples={}
    answers={}
    for arm in ("donor","host"):
        logits=np.load(HERE/"results"/(arm+".npy"))
        assert logits.shape==(768,4) and np.isfinite(logits).all()
        preds=logits.argmax(1)
        answers[arm]=np.bincount(preds,minlength=4).tolist()
        good=preds==gold
        triples[arm]=good.reshape(-1,3).all(1)
        assert int(good.sum())==recorded["individual"][arm]
        assert int(triples[arm].sum())==recorded["complete"][arm]
    d,h=triples["donor"],triples["host"]
    assert np.flatnonzero(d&~h).tolist()==recorded["donor_only"]
    assert np.flatnonzero(h&~d).tolist()==recorded["host_only"]
    delta=d.astype(float)-h
    ix=np.random.default_rng(993613).choice(len(d),size=(5000,len(d)))
    ci=np.quantile(delta[ix].mean(1),[.025,.975])
    np.testing.assert_allclose([delta.mean(),*ci],[recorded["difference"],*recorded["interval"]])
    qualified=(d&~h).sum()>=25 and delta.mean()>=.10 and ci[0]>0
    assert (recorded["decision"]=="advance")==qualified
    print(json.dumps(dict(panel=len(gold),triplets=len(d),donor_complete=int(d.sum()),host_complete=int(h.sum()),
                          donor_only=int((d&~h).sum()),host_only=int((h&~d).sum()),choices=answers,
                          paired_interval=ci.tolist(),qualified=bool(qualified)),indent=2))

if __name__=="__main__":
    main()
