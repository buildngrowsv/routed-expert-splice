"""Check archived full-vocabulary ranks against the pinned two-choice benchmark."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
PRIOR=HERE.parent/"numeric_transfer_study/results"
ARMS=("native","transfer","norm_matched","whole_site")


def paired(scores,full):
    col=0 if full else 3
    native=scores["native"][:,:,col]
    clean=(native==1 if full else native>0).all(1)
    n=int(clean.sum())
    changes={}
    for m in ARMS[1:]:
        if full:
            changes[m]=(scores[m][clean,:,1]==1).all(1)
        else:
            changes[m]=(scores[m][clean,:,3]<0).all(1)
    counts={m:int(v.sum()) for m,v in changes.items()}
    contrast=None
    if n:
        d=changes["transfer"].astype(int)-changes["norm_matched"]
        draw=np.random.default_rng(824625).integers(n,size=(4000,n))
        contrast=dict(mean=float(d.mean()),interval=np.percentile(d[draw].mean(1),[2.5,97.5]).tolist())
    qualified=(n>=100 and min(counts["whole_site"],counts["transfer"])>=.8*n
               and contrast["mean"]>=.2 and contrast["interval"][0]>0)
    return dict(eligible=n,reversals=counts,transfer_minus_random=contrast,qualifies=qualified)


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="1fc65de34e1236722b9fbb4956ee0a7fe71d872f" and receipt["status"]=="done"
    report=json.loads((ROOT/"result.json").read_text())
    original=(PRIOR/"programs.json").read_bytes()
    assert sha256(original).hexdigest()==report["source_sha256"]
    assert sha256((PRIOR/"map.npz").read_bytes()).hexdigest()==report["map_sha256"]
    assert report["groups"]==512
    rows=json.loads(original)["host"]
    own=np.array([r["choices"] for r in rows])
    partner=own[:,::-1]
    with np.load(ROOT/"open-vocabulary.npz") as z:
        assert set(z.files)=={"norms",*ARMS}
        norms=z["norms"]
        assert norms.shape==(1024,3) and np.isfinite(norms).all()
        np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
        scores={m:z[m] for m in ARMS}
    assert all(v.shape==(512,2,4) and np.isfinite(v).all() for v in scores.values())
    evidence={}
    with np.load(PRIOR/"numeric-transfer.npz") as previous:
        for m,v in scores.items():
            assert (v[:,:,:2]>=1).all()
            ranks,top=v[:,:,:2],v[:,:,2]
            assert np.equal(ranks,np.round(ranks)).all() and np.equal(top,np.round(top)).all()
            assert np.array_equal(v[:,:,3]>0,ranks[:,:,0]<ranks[:,:,1])
            assert np.array_equal(v[:,:,3]<0,ranks[:,:,1]<ranks[:,:,0])
            np.testing.assert_array_equal(top[ranks[:,:,0]==1],own[ranks[:,:,0]==1])
            np.testing.assert_array_equal(top[ranks[:,:,1]==1],partner[ranks[:,:,1]==1])
            flips=(v[:,:,3]<0).all(1)
            np.testing.assert_array_equal(flips,(previous[m]<0).all(1))
            evidence[m]=dict(unique_correct_tokens=int((ranks[:,:,0]==1).sum()),
                             unique_partner_tokens=int((ranks[:,:,1]==1).sum()),
                             top_tokens_outside_either_answer=int(((top!=own)&(top!=partner)).sum()))
    forced=paired(scores,False)
    full=paired(scores,True)
    assert forced==report["forced_two_choice"] and full==report["full_vocabulary"]
    assert report["decision"]==("unrestricted_transfer" if full["qualifies"] else "unresolved")
    print(json.dumps(dict(**report,token_counts=evidence),indent=2))


if __name__=="__main__":
    main()
