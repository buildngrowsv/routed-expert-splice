"""Check the first-hop mask screen and its native-correct intersection."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"relay-results"
ARMS=("unrestricted","all_relays","use_only","control_only","without_use")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="37c57362200595f8d323e3d16ae88a253ce93be3"
    raw=json.loads((ROOT/"programs.json").read_text())
    rows=raw["host"]
    assert [r["id"] for r in rows]==list(range(121000,121512))
    assert [r["id"] for r in raw["donor"]]==list(range(122000,122512))
    for h,d in zip(rows,raw["donor"],strict=True):
        a,b=h["relay_spans"]["use"]
        c,e=h["relay_spans"]["control"]
        assert h["site"]<c<e==a<b<=len(h["ids"][0])-1 and b-a==e-c
        assert h["source_values"]==d["source_values"] and h["offset"]!=0 and d["offset"]==0
    report=json.loads((ROOT/"result.json").read_text())
    assert report["earlier_route_sha256"]==sha256((HERE/"results/result.json").read_bytes()).hexdigest()
    assert report["map_sha256"]==sha256((HERE.parent/"numeric_absolute_study/results/map.npz").read_bytes()).hexdigest()
    expected={m+"_"+kind for m in ARMS for kind in ("clean","edited")}
    choices=np.array([r["choices"] for r in rows])
    with np.load(ROOT/"relay.npz") as z:
        assert set(z.files)=={"norms",*expected}
        assert z["norms"].shape==(1024,) and np.isfinite(z["norms"]).all()
        for name in expected:
            a=z[name]
            assert a.shape==(512,2,4) and np.isfinite(a).all()
            own,other=a[:,:,0]==1,a[:,:,1]==1
            np.testing.assert_array_equal(a[:,:,2][own],choices[own])
            np.testing.assert_array_equal(a[:,:,2][other],choices[:,::-1][other])
        clean={m:(z[m+"_clean"][:,:,0]==1).all(1) for m in ARMS}
    eligible=np.logical_and.reduce(tuple(clean.values()))
    assert report["native_by_mask"]=={m:int(v.sum()) for m,v in clean.items()}
    assert report["eligible"]==eligible.sum()==0 and report["decision"]=="unresolved"
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
