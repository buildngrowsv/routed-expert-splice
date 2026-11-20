"""Audit the answer-free scalar, held-out ranks, and exact historical comparisons."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
RAW=HERE.parent/"numeric_raw_32b_study/results"
SCALE=HERE.parent/"numeric_scale_study/results"


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="f00e012434a769932da68c89e832b43644f1b1f4"
    report=json.loads((ROOT/"result.json").read_text())
    cal=json.loads((ROOT/"calibration.json").read_text())
    assert [r["id"] for r in cal]==list(range(123000,123128))
    assert all(r["offset"]==0 and [j for j,(a,b) in enumerate(zip(*r["ids"],strict=True)) if a!=b]==[r["site"]] for r in cal)
    prompts=(SCALE/"programs.json").read_bytes()
    basis=HERE.parent/"numeric_absolute_study/results/map.npz"
    assert sha256(prompts).hexdigest()==report["prompts_digest"]
    assert sha256(basis.read_bytes()).hexdigest()==report["map_digest"]
    choices=np.array([r["choices"] for r in json.loads(prompts)["host"]])
    with np.load(ROOT/"scalar-32b.npz") as z,np.load(RAW/"raw-32b.npz") as raw,np.load(SCALE/"32b.npz") as old:
        assert set(z.files)=={"norms","distances","native","borrowed_raw","norm_matched"}
        doses=z["norms"]
        distances=z["distances"]
        assert distances.shape==(128,2) and np.all(np.isfinite(distances)&(distances>0))
        factor=float(np.median(distances[:,1]/distances[:,0]))
        assert factor==report["gain"]
        assert doses.shape==(1024,2) and np.isfinite(doses).all()
        np.testing.assert_allclose(doses[:,0],factor*raw["norms"][:,0],atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(doses[:,1],doses[:,0],atol=1e-6,rtol=1e-5)
        natives=[(v["native"][:,:,0]==1).all(1) for v in (z,raw,old)]
        assert [int(b.sum()) for b in natives]==report["native_by_run"]
        keep=np.logical_and.reduce(natives)
        n=int(keep.sum())
        assert n==report["eligible"] and n>=100
        for m in ("native","borrowed_raw","norm_matched"):
            a=z[m]
            assert a.shape==(512,2,4) and np.isfinite(a).all()
            assert np.equal(a[:,:,:3],np.round(a[:,:,:3])).all() and (a[:,:,:2]>=1).all()
            first,second=a[:,:,0]==1,a[:,:,1]==1
            np.testing.assert_array_equal(a[:,:,2][first],choices[first])
            np.testing.assert_array_equal(a[:,:,2][second],choices[:,::-1][second])
        arms=dict(gain=z["borrowed_raw"],random=z["norm_matched"],raw=raw["borrowed_raw"],fitted_magnitude=old["borrowed"])
        turned={m:(v[keep,:,1]==1).all(1) for m,v in arms.items()}
        quantities=np.quantile(doses[:,0]/old["norms"][:,3],[.1,.5,.9]).tolist()
    counts={m:int(b.sum()) for m,b in turned.items()}
    assert counts==report["reversals"]
    draws=np.random.default_rng(278042).integers(n,size=(5000,n))
    deltas={}
    for m in ("raw","random","fitted_magnitude"):
        z=turned["gain"].astype(int)-turned[m]
        deltas[m]=dict(mean=float(z.mean()),interval=np.percentile(z[draws].mean(1),[2.5,97.5]).tolist())
    assert deltas==report["gain_minus"]
    passed=(counts["gain"]>=.75*n and deltas["raw"]["interval"][0]>=.2
            and deltas["fitted_magnitude"]["interval"][0]>=-.1 and deltas["random"]["interval"][0]>0)
    assert report["decision"]==("answer_free_gain" if passed else "unresolved")
    print(json.dumps(dict(**report,norm_ratio_to_previous_match=quantities),indent=2))


if __name__=="__main__":
    main()
