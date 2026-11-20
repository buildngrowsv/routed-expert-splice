"""Re-execute the two-class probes and audit matched-path treatment effects."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/"results"
MODES=("unrestricted","direct_only","relay_only")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="51ab60744b7a82b90e7d1a9641ec2b8382eb387a" and receipt["status"]=="done"
    summary=json.loads((ROOT/"result.json").read_text())
    rows=json.loads((ROOT/"programs.json").read_text())
    assert [(r["id"],r["query"]) for r in rows]==[(i,q) for i in range(90000,90512) for q in range(2)]
    for first,second in zip(rows[::2],rows[1::2],strict=True):
        assert first["sites"]==second["sites"] and first["choices"]==second["choices"]
        boundary=first["sites"][-1]+1
        assert all(a[:boundary]==b[:boundary] for a,b in zip(first["ids"],second["ids"],strict=True))
        for row in (first,second):
            assert [j for j,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
            for expected,prompt in enumerate(row["prompts"]):
                for j,c in enumerate(row["letters"]):
                    env={}
                    exec(compile(prompt+c+"\n","<archived attention-path probe>","exec"),env)
                    try:
                        env["fetch"]()
                    except AttributeError:
                        assert expected!=j
                    else:
                        assert expected==j
    with np.load(ROOT/"attention-paths.npz") as data:
        names={f"{m}_{a}" for m in MODES for a in ("clean","edited")}
        assert set(data.files)=={"axis",*names}
        axis=data["axis"].astype(np.float64)
        assert axis.shape==(5120,25)
        np.testing.assert_allclose(axis.T@axis,np.eye(25),atol=1e-6)
        scores={k:data[k] for k in names}
    assert all(v.shape==(512,2,2) and np.isfinite(v).all() for v in scores.values())
    clean={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    cohort=np.logical_and.reduce(tuple(clean.values()))
    n=int(cohort.sum())
    assert summary["eligible"]==n and n>=100
    assert summary["clean_groups"]=={k:int(v.sum()) for k,v in clean.items()}
    flips={m:(scores[m+"_edited"][cohort]<0).all(2) for m in MODES}
    counts={k:v.sum(0).tolist() for k,v in flips.items()}
    assert counts==summary["reversals_by_query"]
    rng=np.random.default_rng(497327)
    sample=rng.integers(n,size=(5000,n))
    def estimate(a,b):
        d=a[:,0].astype(int)-b[:,0]
        return dict(mean=float(d.mean()),interval=np.percentile(d[sample].mean(1),[2.5,97.5]).tolist())
    comparisons={m:estimate(flips[m],flips[MODES[0]]) for m in MODES[1:]}
    assert comparisons==summary["contrast_from_unrestricted"]
    direct=estimate(flips[MODES[1]],flips[MODES[2]])
    assert direct==summary["direct_minus_relay"]
    valid=counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n
    assert valid==summary["unrestricted_control_qualified"]
    sufficient=[m for m in MODES[1:] if valid and counts[m][0]>=.7*n and comparisons[m]["interval"][0]>=-.15]
    lead=("direct_only" if direct["interval"][0]>.20 else "relay_only" if direct["interval"][1]<-.20 else None) if valid else None
    assert summary["sufficient_paths"]==sufficient and summary["more_effective"]==lead
    assert summary["decision"]==("routing_resolved" if sufficient or lead else "unresolved")
    print(json.dumps(dict(**summary,mean_margin={k:v[cohort].mean(axis=(0,2)).tolist() for k,v in scores.items()}),indent=2))


if __name__=="__main__":
    main()
