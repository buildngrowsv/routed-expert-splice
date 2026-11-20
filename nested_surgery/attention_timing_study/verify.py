"""Verify edge-window results and per-group agreement on the saved prompt panel."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/"results"
MODES=("unrestricted","early","late","blocked")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="5040dcc0b6cea204902bcdb4b6fcfd9b10557b56" and receipt["status"]=="done"
    report=json.loads((ROOT/"result.json").read_text())
    assert (report["cut"],report["boundary"],report["depth"])==(10,17,48)
    rows=json.loads((ROOT/"programs.json").read_text())
    assert [(r["id"],r["query"]) for r in rows]==[(i,k) for i in range(98000,98384) for k in range(2)]
    for row in rows:
        assert [j for j,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
        for i,p in enumerate(row["prompts"]):
            for j,c in enumerate(row["letters"]):
                env={}
                exec(compile(p+c+"\n","<archived timing probe>","exec"),env)
                try:
                    env["fetch"]()
                except AttributeError:
                    assert i!=j
                else:
                    assert i==j
    with np.load(ROOT/"timing.npz") as data:
        keys={m+"_"+a for m in MODES for a in ("clean","edited")}
        assert set(data.files)=={"axis",*keys}
        axis=data["axis"].astype(np.float64)
        assert axis.shape==(5120,25)
        np.testing.assert_allclose(axis.T@axis,np.eye(25),atol=1e-6)
        scores={k:data[k] for k in keys}
    assert all(a.shape==(384,2,2) and np.isfinite(a).all() for a in scores.values())
    correct={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    eligible=np.logical_and.reduce(tuple(correct.values()))
    n=int(eligible.sum())
    assert n==report["eligible"] and n>=100
    assert report["clean_groups"]=={m:int(a.sum()) for m,a in correct.items()}
    flip={m:(scores[m+"_edited"][eligible]<0).all(2) for m in MODES}
    counts={m:a.sum(0).tolist() for m,a in flip.items()}
    assert counts==report["reversals"]
    draws=np.random.default_rng(741620).integers(n,size=(5000,n))
    def effect(a,b):
        d=flip[a][:,0].astype(int)-flip[b][:,0]
        return dict(mean=float(d.mean()),interval=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist())
    contrasts={m:effect(m,"unrestricted") for m in ("early","late")}
    assert contrasts==report["relative_to_full"]
    late=effect("late","early")
    assert late==report["late_minus_early"]
    controls=counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n and counts["blocked"][0]<=.15*n
    assert controls==report["controls_qualified"]
    sufficient=[m for m in ("early","late") if controls and counts[m][0]>=.7*n and contrasts[m]["interval"][0]>=-.15]
    lead=("late" if late["interval"][0]>.2 else "early" if late["interval"][1]<-.2 else None) if controls else None
    assert sufficient==report["sufficient"] and lead==report["leading_window"]
    assert report["decision"]==("timing_resolved" if lead or sufficient else "unresolved")
    margins={m:a[eligible].mean(axis=(0,2)).tolist() for m,a in scores.items()}
    exact=int((flip["late"]==flip["unrestricted"]).all(1).sum())
    print(json.dumps(dict(**report,mean_margin=margins,late_full_agree_in_both_queries=exact),indent=2))


if __name__=="__main__":
    main()
