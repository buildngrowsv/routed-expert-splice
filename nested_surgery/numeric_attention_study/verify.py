"""Reconstruct unrestricted digit outcomes across frozen identifier-selected masks."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
MODES=("unrestricted","blocked","strong_eight","weak_eight","without_strong")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="2a11706fb2ee80eb9394f3dab5a003e5a706512f"
    report=json.loads((ROOT/"result.json").read_text())
    basis=HERE.parent/"numeric_absolute_study/results/map.npz"
    route=HERE.parent/"attention_sparse_study/results/result.json"
    assert report["map_sha256"]==sha256(basis.read_bytes()).hexdigest()
    assert report["route_sha256"]==sha256(route.read_bytes()).hexdigest()
    original=json.loads(route.read_text())
    assert report["strong"]==original["strong"] and report["weak"]==original["weak"]
    raw=json.loads((ROOT/"programs.json").read_text())
    assert [r["id"] for r in raw["host"]]==list(range(119000,119512))
    assert [r["id"] for r in raw["donor"]]==list(range(120000,120512))
    for h,d in zip(raw["host"],raw["donor"],strict=True):
        assert h["source_values"]==d["source_values"] and h["offset"]!=0 and d["offset"]==0
        assert set(map(int,h["answers"])).isdisjoint(d["source_values"])
        stem=re.search(r"amount_\d+",d["prompts"][0]).group()
        assert all(stem not in p for p in h["prompts"])
        for row in (h,d):
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==[row["site"]]
            for i,p in enumerate(row["prompts"]):
                for j,c in enumerate(row["answers"]):
                    scope={}
                    try:
                        exec(compile(p+c+"\n","<archived route probe>","exec"),scope)
                    except AssertionError:
                        assert i!=j
                    else:
                        assert i==j and scope["fetch"]()==int(c)
    names={m+"_"+a for m in MODES for a in ("clean","edited")}
    choices=np.array([r["choices"] for r in raw["host"]])
    with np.load(ROOT/"numeric-attention.npz") as z:
        assert set(z.files)=={"norms",*names}
        doses=z["norms"]
        assert doses.shape==(1024,) and np.isfinite(doses).all()
        scores={k:z[k] for k in names}
    for a in scores.values():
        assert a.shape==(512,2,4) and np.isfinite(a).all()
        own,other=a[:,:,0]==1,a[:,:,1]==1
        np.testing.assert_array_equal(a[:,:,2][own],choices[own])
        np.testing.assert_array_equal(a[:,:,2][other],choices[:,::-1][other])
        np.testing.assert_array_equal(a[:,:,0]<a[:,:,1],a[:,:,3]>0)
    native={m:(scores[m+"_clean"][:,:,0]==1).all(1) for m in MODES}
    common=np.logical_and.reduce(tuple(native.values()))
    n=int(common.sum())
    assert n==report["eligible"] and n>=100
    assert {m:int(v.sum()) for m,v in native.items()}==report["clean_groups"]
    flips={m:(scores[m+"_edited"][common,:,1]==1).all(1) for m in MODES}
    count={m:int(v.sum()) for m,v in flips.items()}
    assert count==report["reversals"]
    draws=np.random.default_rng(213816).integers(n,size=(5000,n))
    def contrast(a,b):
        d=flips[a].astype(int)-flips[b]
        return dict(mean=float(d.mean()),interval=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist())
    high=contrast("strong_eight","unrestricted")
    low=contrast("strong_eight","weak_eight")
    removed=contrast("unrestricted","without_strong")
    assert high==report["strong_minus_full"] and low==report["strong_minus_weak"] and removed==report["removal"]
    control=count["unrestricted"]>=.8*n
    passed=(control and count["blocked"]<=.15*n and count["strong_eight"]>=.7*n
            and high["interval"][0]>=-.15 and low["interval"][0]>.2 and removed["interval"][0]>.2)
    assert control==report["unrestricted_control_qualified"]
    assert report["decision"]==("route_transfers" if passed else "unresolved")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
