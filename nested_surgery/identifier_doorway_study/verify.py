"""Recompute the relay controls and paired comparisons from the public archive."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent/"results"
ARMS = ("unrestricted","same_class_relay","other_class_relay")


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="addc988df65c36b6ed09cb49f91f48efbbc06269" and receipt["status"]=="done"
    report = json.loads((ROOT/"result.json").read_text())
    assert report["cut"]==10 and report["relay_blocks"]==7
    rows = json.loads((ROOT/"programs.json").read_text())
    assert [(r["id"],r["query"]) for r in rows]==[(i,q) for i in range(32000,32256) for q in range(2)]
    for row in rows:
        assert row["relays"]==[s+1 for s in row["sites"]]
        assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
        assert [row["ids"][0][i] for i in row["relays"]]==[row["ids"][1][i] for i in row["relays"]]
    with np.load(ROOT/"doorway.npz") as data:
        assert set(data.files)=={"native","relay_release_norms",*ARMS}
        norms=data["relay_release_norms"]
        assert norms.shape==(256,2,2,2) and np.isfinite(norms).all()
        values={k:data[k] for k in ("native",*ARMS)}
    assert all(a.shape==(256,2,2) and np.isfinite(a).all() for a in values.values())
    clean=(values["native"]>0).all(axis=(1,2))
    n=int(clean.sum())
    assert n==report["eligible_groups"] and n>=100
    draws=np.random.default_rng(528092).integers(n,size=(5000,n))
    def estimate(a,b):
        x=a.astype(int)-b
        return dict(mean=float(x.mean()),interval=np.percentile(x[draws].mean(1),[2.5,97.5]).tolist())
    flips={arm:(values[arm][clean]<0).all(2) for arm in ARMS}
    counts={k:dict(first=int(v[:,0].sum()),second=int(v[:,1].sum()),differential=estimate(v[:,0],v[:,1]))
            for k,v in flips.items()}
    assert counts==report["reversals"]
    advantage=estimate(flips[ARMS[1]][:,0],flips[ARMS[2]][:,0])
    assert advantage==report["relay_advantage"]
    medians=np.median(norms,axis=(0,1,2)).tolist()
    assert medians==report["median_release_norm"]
    free=counts["unrestricted"]
    ready=free["first"]>=.6*n and free["second"]<=.15*n
    qualifying=[k for k in ARMS[1:] if ready and counts[k]["first"]>=.35*n
                and counts[k]["differential"]["interval"][0]>0]
    assert qualifying==report["qualifying_relays"]
    assert report["decision"]==("single_relay" if qualifying else "unresolved")
    margins={k:v[clean].mean(axis=(0,2)).tolist() for k,v in values.items()}
    print(json.dumps(dict(**report,mean_original_letter_margin=margins),indent=2))


if __name__=="__main__":
    main()
