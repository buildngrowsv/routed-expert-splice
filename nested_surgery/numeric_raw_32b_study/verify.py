"""Audit the raw 32B edit against the exact earlier calibrated-magnitude result."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
SCALE=HERE.parent/"numeric_scale_study/results"
MODES=("native","borrowed_raw","norm_matched")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="92686641e3d180b0d1ffcd86359688b39eedf4ec"
    report=json.loads((ROOT/"result.json").read_text())
    programs=(SCALE/"programs.json").read_bytes()
    assert report["prompts_digest"]==sha256(programs).hexdigest()
    source=HERE.parent/"numeric_absolute_study/results/map.npz"
    assert report["map_digest"]==sha256(source.read_bytes()).hexdigest()
    rows=json.loads(programs)["host"]
    choices=np.array([r["choices"] for r in rows])
    with np.load(ROOT/"raw-32b.npz") as z,np.load(SCALE/"32b.npz") as old:
        assert set(z.files)=={"norms",*MODES}
        norms=z["norms"]
        assert norms.shape==(1024,2) and np.isfinite(norms).all()
        np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
        ratios=norms[:,0]/old["norms"][:,3]
        assert np.quantile(ratios,[.1,.5,.9]).tolist()==report["norm_ratio_quantiles"]
        data={m:z[m] for m in MODES}
        prior={m:old[m] for m in ("native","borrowed")}
    for v in data.values():
        assert v.shape==(512,2,4) and np.isfinite(v).all()
        assert (v[:,:,:2]>=1).all() and np.equal(v[:,:,:3],np.round(v[:,:,:3])).all()
        np.testing.assert_array_equal(v[:,:,0]<v[:,:,1],v[:,:,3]>0)
        own,other=v[:,:,0]==1,v[:,:,1]==1
        np.testing.assert_array_equal(v[:,:,2][own],choices[own])
        np.testing.assert_array_equal(v[:,:,2][other],choices[:,::-1][other])
    now=(data["native"][:,:,0]==1).all(1)
    past=(prior["native"][:,:,0]==1).all(1)
    selected=now&past
    n=int(selected.sum())
    assert report["native_new"]==now.sum() and report["native_prior"]==past.sum() and report["eligible"]==n
    np.testing.assert_array_equal(now,past)
    switches={k:(v[selected,:,1]==1).all(1) for k,v in data.items() if k!="native"}
    switches["borrowed_scaled"]=(prior["borrowed"][selected,:,1]==1).all(1)
    counts={m:int(a.sum()) for m,a in switches.items()}
    assert counts==report["reversals"]
    draws=np.random.default_rng(640393).integers(n,size=(5000,n))
    def contrast(reference):
        values=switches["borrowed_raw"].astype(int)-switches[reference]
        return dict(mean=float(values.mean()),interval=np.percentile(values[draws].mean(1),[2.5,97.5]).tolist())
    random=contrast("norm_matched")
    scaled=contrast("borrowed_scaled")
    assert random==report["raw_minus_random"] and scaled==report["raw_minus_scaled"]
    qualified=n>=100 and counts["borrowed_raw"]>=.75*n and random["mean"]>=.20 and random["interval"][0]>0
    assert report["decision"]==("unscaled_transfer" if qualified else "unresolved")
    assert report["near_scaled"]==bool(scaled["interval"][0]>=-.10)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
