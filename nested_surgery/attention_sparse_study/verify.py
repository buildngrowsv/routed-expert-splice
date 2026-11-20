"""Audit exact selection bins, clean retention and paired held-out edge contrasts."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
MODES=("unrestricted","blocked","strong_eight","weak_eight","without_strong")


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="7f8d9bdd92d2e3ab0e918f213a0d29c79507f4f3"
    result=json.loads((ROOT/"result.json").read_text())
    earlier=HERE.parent/"attention_single_study/results/single-edge.npz"
    assert sha256(earlier.read_bytes()).hexdigest()==result["calibration_digest"]
    rows=json.loads((ROOT/"programs.json").read_text())
    assert [(r["id"],r["query"]) for r in rows]==[(i,j) for i in range(116000,116384) for j in range(2)]
    for r in rows:
        assert [i for i,(a,b) in enumerate(zip(*r["ids"],strict=True)) if a!=b]==r["sites"]
        for i,prompt in enumerate(r["prompts"]):
            for j,c in enumerate(r["letters"]):
                env={}
                exec(compile(prompt+c+"\n","<archived sparse-edge probe>","exec"),env)
                try:
                    env["fetch"]()
                except AttributeError:
                    assert i!=j
                else:
                    assert i==j
    names={m+"_"+a for m in MODES for a in ("clean","edited")}
    with np.load(ROOT/"sparse-edges.npz") as z,np.load(earlier) as old:
        assert set(z.files)=={"axis","selection_strength",*names}
        a=z["axis"].astype(np.float64)
        assert a.shape==(5120,25)
        np.testing.assert_allclose(a.T@a,np.eye(25),atol=1e-6)
        strength=z["selection_strength"]
        np.testing.assert_array_equal(strength,old["selection_strength"])
        scores={k:z[k] for k in names}
    high,low=[],[]
    for left,right in ((18,26),(26,34),(34,42),(42,49)):
        p=np.arange(left,right)
        a=p[np.lexsort((p,-strength[p-18]))][:2]
        remaining=np.setdiff1d(p,a)
        b=remaining[np.lexsort((remaining,strength[remaining-18]))][:2]
        high.extend(a)
        low.extend(b)
    assert sorted(high)==result["strong"] and sorted(low)==result["weak"]
    assert len(set(high+low))==16
    assert all(v.shape==(384,2,2) and np.isfinite(v).all() for v in scores.values())
    clean={m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    eligible=np.logical_and.reduce(tuple(clean.values()))
    n=int(eligible.sum())
    assert n==result["eligible"] and n>=100
    assert {m:int(v.sum()) for m,v in clean.items()}==result["clean_groups"]
    flips={m:(scores[m+"_edited"][eligible]<0).all(2) for m in MODES}
    counts={m:v.sum(0).tolist() for m,v in flips.items()}
    assert counts==result["reversals"]
    draws=np.random.default_rng(416083).integers(n,size=(5000,n))
    def delta(a,b):
        d=flips[a][:,0].astype(int)-flips[b][:,0]
        return dict(mean=float(d.mean()),interval=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist())
    relative={m:delta(m,"unrestricted") for m in ("strong_eight","weak_eight")}
    high_low=delta("strong_eight","weak_eight")
    removed=delta("unrestricted","without_strong")
    assert relative==result["relative"] and high_low==result["strong_minus_weak"] and removed==result["removal"]
    control=counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n and counts["blocked"][0]<=.15*n
    sufficient=[m for m,d in relative.items() if control and counts[m][0]>=.7*n and d["interval"][0]>=-.15]
    lead=control and high_low["interval"][0]>=.2
    needed=control and removed["interval"][0]>=.2
    assert control==result["controls_qualified"] and sufficient==result["sufficient"]
    assert lead==result["strong_leads"] and needed==result["removal_material"]
    assert result["decision"]==("sparse_attention_resolved" if sufficient or lead or needed else "unresolved")
    marg={m:a[eligible].mean(axis=(0,2)).tolist() for m,a in scores.items()}
    overlap=int((flips["unrestricted"][:,0]&flips["strong_eight"][:,0]).sum())
    print(json.dumps(dict(**result,mean_margins=marg,overlap_with_full=overlap),indent=2))


if __name__=="__main__":
    main()
