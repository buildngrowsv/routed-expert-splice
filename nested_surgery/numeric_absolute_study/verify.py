"""Verify the absolute map, executable programs and exact full-vocabulary outcome."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parent/"results"
MODES=("native","transfer","norm_matched","paired","whole_site")


def metric(data,full):
    natively=data["native"][:,:,0 if full else 3]
    good=(natively==1 if full else natively>0).all(1)
    n=int(good.sum())
    turns={}
    for m in MODES[1:]:
        values=data[m][good,:,1 if full else 3]
        turns[m]=(values==1 if full else values<0).all(1)
    totals={m:int(v.sum()) for m,v in turns.items()}
    difference=None
    if n:
        delta=turns["transfer"].astype(int)-turns["norm_matched"]
        idx=np.random.default_rng(824625).integers(n,size=(4000,n))
        difference=dict(mean=float(delta.mean()),interval=np.percentile(delta[idx].mean(1),[2.5,97.5]).tolist())
    qualified=(n>=100 and min(totals["transfer"],totals["whole_site"])>=.8*n
               and difference["mean"]>=.2 and difference["interval"][0]>0)
    return dict(eligible=n,reversals=totals,transfer_minus_random=difference,qualifies=qualified),turns


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="e29adff07c4d132863e855b93d1319ed5ebed808"
    result=json.loads((ROOT/"result.json").read_text())
    rows=json.loads((ROOT/"programs.json").read_text())
    assert [r["id"] for r in rows["host"]]==list(range(114000,114512))
    assert [r["id"] for r in rows["donor"]]==list(range(115000,115512))
    for h,d in zip(rows["host"],rows["donor"],strict=True):
        assert h["source_values"]==d["source_values"] and h["offset"]!=0 and d["offset"]==0
        assert set(map(int,h["answers"])).isdisjoint(d["source_values"])
        donor_stem=re.search(r"amount_\d+",d["prompts"][0]).group()
        assert all(donor_stem not in p for p in h["prompts"])
        for row in (h,d):
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==[row["site"]]
            for expected,prompt in enumerate(row["prompts"]):
                for j,answer in enumerate(row["answers"]):
                    ns={}
                    try:
                        exec(compile(prompt+answer+"\n","<archived one-pass operand>","exec"),ns)
                    except AssertionError:
                        assert j!=expected
                    else:
                        assert j==expected and ns["fetch"]()==int(answer)
    with np.load(ROOT/"map.npz") as maps:
        shape=dict(donor=(3584,7),host=(5120,7),orthogonal=(5120,7),paired_map=(7,7),absolute_map=(7,7),bias=(7,))
        assert set(maps.files)==set(shape)
        assert all(maps[k].shape==s and np.isfinite(maps[k]).all() for k,s in shape.items())
        for k in ("donor","host","orthogonal"):
            axes=maps[k].astype(np.float64)
            np.testing.assert_allclose(axes.T@axes,np.eye(7),atol=1e-6)
        np.testing.assert_allclose(maps["host"].astype(np.float64).T@maps["orthogonal"],0,atol=1e-6)
    choices=np.array([r["choices"] for r in rows["host"]])
    with np.load(ROOT/"absolute.npz") as z:
        assert set(z.files)=={"norms",*MODES}
        norms=z["norms"]
        assert norms.shape==(1024,3) and np.isfinite(norms).all()
        np.testing.assert_allclose(norms[:,0],norms[:,1],atol=1e-6,rtol=1e-5)
        scores={m:z[m] for m in MODES}
    for v in scores.values():
        assert v.shape==(512,2,4) and np.isfinite(v).all()
        assert (v[:,:,:2]>=1).all() and np.equal(v[:,:,:3],np.round(v[:,:,:3])).all()
        np.testing.assert_array_equal(v[:,:,0]<v[:,:,1],v[:,:,3]>0)
        np.testing.assert_array_equal(v[:,:,1]<v[:,:,0],v[:,:,3]<0)
        own,other=v[:,:,0]==1,v[:,:,1]==1
        np.testing.assert_array_equal(v[:,:,2][own],choices[own])
        np.testing.assert_array_equal(v[:,:,2][other],choices[:,::-1][other])
    unrestricted,turns=metric(scores,True)
    forced,_=metric(scores,False)
    assert result["full_vocabulary"]==unrestricted and result["forced_two_choice"]==forced
    passed=unrestricted["qualifies"] and unrestricted["reversals"]["paired"]>=.8*unrestricted["eligible"]
    assert result["decision"]==("one_pass_transfer" if passed else "unresolved")
    print(json.dumps(dict(**result,agreeing_pairs=int((turns["transfer"]==turns["paired"]).sum()),
                          map_sha256=sha256((ROOT/"map.npz").read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":
    main()
