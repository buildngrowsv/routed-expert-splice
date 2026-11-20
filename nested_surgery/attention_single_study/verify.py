"""Audit the preselected block and its held-out necessity and sufficiency tests."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/"results"
MODES=("unrestricted","blocked","only_selected","except_selected")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="3e428e7a2e955398056e38a56193e42f4877b579" and receipt["status"]=="done"
    report=json.loads((ROOT/"result.json").read_text())
    panels=json.loads((ROOT/"programs.json").read_text())
    for key,start,count in (("selection",104000,64),("test",105000,384)):
        rows=panels[key]
        assert [(r["id"],r["query"]) for r in rows]==[(i,q) for i in range(start,start+count) for q in range(2)]
        for row in rows:
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
            for i,p in enumerate(row["prompts"]):
                for j,c in enumerate(row["letters"]):
                    ns={}
                    exec(compile(p+c+"\n","<archived selected-edge probe>","exec"),ns)
                    try:
                        ns["fetch"]()
                    except AttributeError:
                        assert i!=j
                    else:
                        assert i==j
    with np.load(ROOT/"single-edge.npz") as z:
        names={m+"_"+a for m in MODES for a in ("clean","edited")}
        assert set(z.files)=={"axis","selection_strength",*names}
        fit=z["axis"].astype(np.float64)
        assert fit.shape==(5120,25)
        np.testing.assert_allclose(fit.T@fit,np.eye(25),atol=1e-6)
        strengths=z["selection_strength"]
        assert strengths.shape==(31,) and np.isfinite(strengths).all() and np.all(strengths>=0)
        assert report["selected_block"]==int(strengths.argmax())+18
        scores={k:z[k] for k in names}
    assert all(a.shape==(384,2,2) and np.isfinite(a).all() for a in scores.values())
    clean={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    eligible=np.logical_and.reduce(tuple(clean.values()))
    n=int(eligible.sum())
    assert report["eligible"]==n and n>=100
    assert report["clean_groups"]=={m:int(a.sum()) for m,a in clean.items()}
    flips={m:(scores[m+"_edited"][eligible]<0).all(2) for m in MODES}
    counts={m:a.sum(0).tolist() for m,a in flips.items()}
    assert counts==report["reversals"]
    rng=np.random.default_rng(107481)
    draws=rng.integers(n,size=(5000,n))
    def compare(a,b):
        d=flips[a][:,0].astype(int)-flips[b][:,0]
        return dict(mean=float(d.mean()),interval=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist())
    only=compare("only_selected","unrestricted")
    removal=compare("unrestricted","except_selected")
    assert only==report["only_margin"] and removal==report["removal_effect"]
    controls=counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n and counts["blocked"][0]<=.15*n
    assert controls==report["controls_qualified"]
    sufficient=controls and counts["only_selected"][0]>=.7*n and only["interval"][0]>=-.15
    necessary=controls and removal["interval"][0]>=.2
    assert sufficient==report["sufficient"] and necessary==report["necessary"]
    assert report["decision"]==("isolated_edge" if sufficient or necessary else "unresolved")
    out=dict(**report,calibration_strengths=strengths.tolist(),
             mean_margins={m:a[eligible].mean(axis=(0,2)).tolist() for m,a in scores.items()})
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
