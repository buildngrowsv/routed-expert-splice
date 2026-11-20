"""Verify the full-vocabulary composition endpoint against the exact prior panel."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
PRIOR=HERE.parent/"numeric_simple_study/results"
ARMS=("native","both","random","whole")


def compare(scores,raw,*,full):
    states=np.arange(4)
    win={}
    for k,v in scores.items():
        expected=(states if k=="native" else states[::-1])[None,:,None]
        if full:
            success=np.take_along_axis(v[:,:,4:8],expected,axis=2)[...,0]==1
        else:
            alternatives=v[:,:,:4]
            best=np.take_along_axis(alternatives,expected,axis=2)[...,0]
            remainder=np.max(np.where(states[None,None,:]==expected,-np.inf,alternatives),axis=2)
            success=best>remainder
        win[k]=success.all(1)
    clean=win["native"]
    n=int(clean.sum())
    cells={k:int(a[clean].sum()) for k,a in win.items()}
    grouped={}
    signs=np.array([r["sign"] for r in raw])
    for op,s in (("addition",1),("subtraction",-1)):
        ok=clean&(signs==s)
        grouped[op]=dict(eligible=int(ok.sum()),correct={k:int(a[ok].sum()) for k,a in win.items()})
    contrast=None
    if n:
        d=win["both"][clean].astype(int)-win["random"][clean]
        sample=np.random.default_rng(214790).integers(n,size=(5000,n))
        contrast=dict(mean=float(d.mean()),interval=np.percentile(d[sample].mean(1),[2.5,97.5]).tolist())
    qualified=(n>=100 and min(cells["both"],cells["whole"])>=.8*n
               and contrast["mean"]>=.2 and contrast["interval"][0]>0)
    return dict(eligible=n,correct=cells,both_minus_random=contrast,qualifies=qualified,by_operator=grouped)


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="0305bacfd0837e0ed6d7b15c781fc57ab3dca28a" and receipt["status"]=="done"
    record=json.loads((ROOT/"result.json").read_text())
    previous=(PRIOR/"programs.json").read_bytes()
    source_map=HERE.parent/"numeric_transfer_study/results/map.npz"
    assert sha256(previous).hexdigest()==record["source_digest"]
    assert sha256(source_map.read_bytes()).hexdigest()==record["map_digest"]
    rows=json.loads(previous)["host"]
    assert len(rows)==record["groups"]==512
    correct_ids=np.array([r["choices"] for r in rows])
    with np.load(ROOT/"composition-open.npz") as now, np.load(PRIOR/"composition.npz") as past:
        assert set(now.files)=={*ARMS,"left_norms","right_norms"}
        for side in ("left_norms","right_norms"):
            a=now[side]
            assert a.shape==(512,2,2) and np.isfinite(a).all()
            np.testing.assert_allclose(a[...,0],a[...,1],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(a,past[side],rtol=1e-5,atol=1e-6)
        scores={m:now[m] for m in ARMS}
        outside={}
        for m,v in scores.items():
            assert v.shape==(512,4,9) and np.isfinite(v).all()
            z,ranks,top=v[:,:,:4],v[:,:,4:8],v[:,:,8]
            assert np.all(ranks>=1) and np.equal(ranks,np.round(ranks)).all()
            assert np.equal(top,np.round(top)).all()
            np.testing.assert_array_equal(z.argmax(2),past[m].argmax(2))
            np.testing.assert_array_equal(z[:,:,:,None]>z[:,:,None,:],ranks[:,:,:,None]<ranks[:,:,None,:])
            g,t,c=np.where(ranks==1)
            np.testing.assert_array_equal(top[g,t],correct_ids[g,c])
            outside[m]=int((top[:,:,None]!=correct_ids[:,None,:]).all(2).sum())
    full=compare(scores,rows,full=True)
    forced=compare(scores,rows,full=False)
    assert full==record["full_vocabulary"] and forced==record["forced_four_choice"]
    assert record["decision"]==("composition_unrestricted" if full["qualifies"] else "unresolved")
    print(json.dumps(dict(**record,top_token_outside_choices=outside),indent=2))


if __name__=="__main__":
    main()
