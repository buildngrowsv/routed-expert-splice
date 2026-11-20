"""Re-execute calibration and arithmetic examples, then audit the paired effects."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent/"results"


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="f6373b266767c5d226ea86c036471f719e7e0819"
    report=json.loads((ROOT/"result.json").read_text())
    panels=json.loads((ROOT/"programs.json").read_text())
    for name,start in (("calibration",0),("test",60000)):
        rows=panels[name]
        assert [r["id"] for r in rows]==list(range(start,start+512))
        for r in rows:
            offset=r["offset"]
            assert (offset==0)==(name=="calibration")
            if offset:
                assert set(map(int,r["answers"])).isdisjoint(r["source_values"])
            assert [j for j,(a,b) in enumerate(zip(*r["ids"],strict=True)) if a!=b]==[r["site"]]
            for i,prompt in enumerate(r["prompts"]):
                for j,answer in enumerate(r["answers"]):
                    ns={}
                    try:
                        exec(compile(prompt+answer+"\n","<archived arithmetic probe>","exec"),ns)
                    except AssertionError:
                        assert i!=j
                    else:
                        assert i==j and ns["fetch"]()==int(answer)==r["source_values"][i]+offset
    with np.load(ROOT/"computed.npz") as z:
        assert set(z.files)=={"norms","native","learned","matched","whole_site"}
        norms=z["norms"]
        assert norms.shape==(1024,3) and np.isfinite(norms).all()
        np.testing.assert_allclose(norms[:,1],norms[:,2],rtol=1e-5,atol=1e-6)
        scores={k:z[k] for k in ("native","learned","matched","whole_site")}
    assert all(v.shape==(512,2) and np.isfinite(v).all() for v in scores.values())
    eligible=(scores["native"]>0).all(1)
    n=int(eligible.sum())
    assert n==report["arm"]["eligible"] and n>=100
    flipped={k:(v[eligible]<0).all(1) for k,v in scores.items() if k!="native"}
    counts={k:int(v.sum()) for k,v in flipped.items()}
    assert report["arm"]["switches"]==counts
    diff=flipped["learned"].astype(int)-flipped["matched"]
    idx=np.random.default_rng(50324).integers(n,size=(5000,n))
    interval=dict(mean=float(diff.mean()),interval=np.percentile(diff[idx].mean(1),[2.5,97.5]).tolist())
    assert report["arm"]["learned_minus_matched"]==interval
    passed=counts["learned"]>=.6*n and counts["whole_site"]>=.6*n and diff.mean()>=.2 and interval["interval"][0]>0
    assert report["decision"]==("computed_transfer" if passed else "unresolved")
    offsets=np.asarray([r["offset"] for r in panels["test"]])
    splits={}
    for name,mask in (("addition",offsets>0),("subtraction",offsets<0)):
        base=eligible&mask
        splits[name]=dict(eligible=int(base.sum()),reversals={k:int((v[base]<0).all(1).sum()) for k,v in scores.items()
                                                             if k!="native"})
    print(json.dumps(dict(**report,by_operator=splits,mean_margin={k:float(v[eligible].mean()) for k,v in scores.items()},
                          max_norm_difference=float(abs(norms[:,1]-norms[:,2]).max())),indent=2))


if __name__=="__main__":
    main()
