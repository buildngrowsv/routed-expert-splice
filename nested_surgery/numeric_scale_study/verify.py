"""Reconstruct 14B and 32B native cohorts, edits, and same-norm interface contrasts."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
MODES=("native","transfer","norm_matched","paired","whole_site")


def compare(scores,full):
    clean=(scores["native"][:,:,0]==1 if full else scores["native"][:,:,3]>0).all(1)
    n=int(clean.sum())
    outcome={}
    for arm,z in scores.items():
        if arm=="native":
            continue
        outcome[arm]=(z[clean,:,1]==1 if full else z[clean,:,3]<0).all(1)
    counts={a:int(x.sum()) for a,x in outcome.items()}
    effect=None
    if n:
        d=outcome["transfer"].astype(int)-outcome["norm_matched"]
        indices=np.random.default_rng(824625).integers(n,size=(4000,n))
        effect=dict(mean=float(d.mean()),interval=np.percentile(d[indices].mean(1),[2.5,97.5]).tolist())
    qualified=(n>=100 and min(counts["transfer"],counts["whole_site"])>=.8*n
               and effect["mean"]>=.2 and effect["interval"][0]>0)
    return dict(eligible=n,reversals=counts,transfer_minus_random=effect,qualifies=qualified),clean,outcome


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="ea4014f0edf250fa8abf5d0bb52657371998aea3"
    result=json.loads((ROOT/"result.json").read_text())
    old=HERE.parent/"numeric_absolute_study/results/map.npz"
    assert sha256(old.read_bytes()).hexdigest()==result["old_map_sha256"]
    data=json.loads((ROOT/"programs.json").read_text())
    assert [r["id"] for r in data["host"]]==list(range(117000,117512))
    assert [r["id"] for r in data["donor"]]==list(range(118000,118512))
    for h,d in zip(data["host"],data["donor"],strict=True):
        assert h["source_values"]==d["source_values"] and h["offset"]!=0 and d["offset"]==0
        assert set(map(int,h["answers"])).isdisjoint(d["source_values"])
        stem=re.search(r"amount_\d+",d["prompts"][0]).group()
        assert all(stem not in p for p in h["prompts"])
        for row in (h,d):
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==[row["site"]]
            for i,p in enumerate(row["prompts"]):
                for j,x in enumerate(row["answers"]):
                    env={}
                    try:
                        exec(compile(p+x+"\n","<archived receiver probe>","exec"),env)
                    except AssertionError:
                        assert i!=j
                    else:
                        assert i==j and env["fetch"]()==int(x)
    with np.load(ROOT/"map-32b.npz") as z:
        shape=dict(donor=(3584,7),host=(5120,7),orthogonal=(5120,7),paired=(7,7),absolute=(7,7),bias=(7,))
        assert set(z.files)==set(shape) and all(z[k].shape==s and np.isfinite(z[k]).all() for k,s in shape.items())
        for k in ("donor","host","orthogonal"):
            a=z[k].astype(np.float64)
            np.testing.assert_allclose(a.T@a,np.eye(7),atol=1e-6)
        np.testing.assert_allclose(z["host"].astype(np.float64).T@z["orthogonal"],0,atol=1e-6)
    choices=np.array([r["choices"] for r in data["host"]])
    receiver_data,clean,turns={}, {}, {}
    for model,arm_count in (("14b",3),("32b",4)):
        arms=MODES if model=="14b" else (*MODES,"borrowed")
        with np.load(ROOT/(model+".npz")) as z:
            assert set(z.files)=={*arms,"norms"}
            norms=z["norms"]
            assert norms.shape==(1024,arm_count) and np.isfinite(norms).all()
            np.testing.assert_allclose(norms[:,0],norms[:,1],atol=1e-6,rtol=1e-5)
            if model=="32b":
                np.testing.assert_allclose(norms[:,0],norms[:,3],atol=1e-6,rtol=1e-5)
            scores={k:z[k] for k in arms}
        for v in scores.values():
            assert v.shape==(512,2,4) and np.isfinite(v).all()
            assert (v[:,:,:2]>=1).all() and np.equal(v[:,:,:3],np.round(v[:,:,:3])).all()
            np.testing.assert_array_equal(v[:,:,0]<v[:,:,1],v[:,:,3]>0)
            np.testing.assert_array_equal(v[:,:,0]>v[:,:,1],v[:,:,3]<0)
            own,partner=v[:,:,0]==1,v[:,:,1]==1
            np.testing.assert_array_equal(v[:,:,2][own],choices[own])
            np.testing.assert_array_equal(v[:,:,2][partner],choices[:,::-1][partner])
        full,clean[model],turns[model]=compare(scores,True)
        restricted,_,_=compare(scores,False)
        assert result["arms"][model]==dict(full_vocabulary=full,forced=restricted)
        receiver_data[model]=scores
    both=clean["14b"]&clean["32b"]
    shared={m:{k:int((a[both,:,1]==1).all(1).sum()) for k,a in scores.items() if k!="native"}
            for m,scores in receiver_data.items()}
    assert result["shared"]==dict(eligible=int(both.sum()),arms=shared)
    n=int(clean["32b"].sum())
    contrast=None
    if n:
        d=turns["32b"]["transfer"].astype(int)-turns["32b"]["borrowed"]
        sample=np.random.default_rng(501306).integers(n,size=(5000,n))
        contrast=dict(mean=float(d.mean()),interval=np.percentile(d[sample].mean(1),[2.5,97.5]).tolist())
    assert result["fitted_minus_borrowed"]==contrast
    assert result["fit_advantage"]==bool(contrast and contrast["interval"][0]>=.2)
    passed=result["arms"]["32b"]["full_vocabulary"]["qualifies"] and turns["32b"]["paired"].sum()>=.8*n
    assert result["decision"]==("scales_to_32b" if passed else "unresolved")
    with np.load(old) as prior, np.load(ROOT/"map-32b.npz") as fresh:
        u=prior["host"].astype(np.float64)
        alignment={}
        for name,k in (("learned_32b","host"),("random_32b","orthogonal")):
            angles=np.linalg.svd(u.T@fresh[k].astype(np.float64),compute_uv=False)
            alignment[name]=dict(principal_cosines=angles.tolist(),mean_squared_cosine=float(np.mean(angles**2)))
    print(json.dumps(dict(**result,exploratory_alignment=alignment),indent=2))


if __name__=="__main__":
    main()
