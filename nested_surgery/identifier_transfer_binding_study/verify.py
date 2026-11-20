"""Audit the independent source contexts, map digest, doses and paired outcomes."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
STEM=re.compile(r"item_\d+")


def read(prompt,letter):
    env={}
    exec(compile(prompt+letter+"\n","<archived generated transfer probe>","exec"),env)
    return env["fetch"]()


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="3d0030edb7e7a778abe06400719b9b4335d51a5c" and receipt["status"]=="done"
    report=json.loads((ROOT/"result.json").read_text())
    saved_map=HERE.parent/"identifier_transfer_study/results/map.npz"
    assert sha256(saved_map.read_bytes()).hexdigest()==report["map_digest"]
    data=json.loads((ROOT/"programs.json").read_text())
    assert [r["id"] for r in data["donor"]]==list(range(50000,50256))
    assert [(r["id"],r["query"]) for r in data["host"]]==[(i,q) for i in range(40000,40256) for q in range(2)]
    for donor,first,second in zip(data["donor"],data["host"][::2],data["host"][1::2],strict=True):
        assert donor["letters"]==first["letters"]==second["letters"]
        stem=STEM.search(donor["prompts"][0]).group()
        assert all(stem not in p for row in (first,second) for p in row["prompts"])
        assert [read(p,c) for p,c in zip(donor["prompts"],donor["letters"],strict=True)]==[7,7]
        for row in (first,second):
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
            answers=[read(p,c) for p,c in zip(row["prompts"],row["letters"],strict=True)]
            assert answers[0]==answers[1]
    with np.load(ROOT/"cross-binding.npz") as z:
        assert set(z.files)=={"perturbation_norms","native","selected","unselected","whole_site"}
        norms=z["perturbation_norms"]
        assert norms.shape==(256,2,2,2) and np.isfinite(norms).all()
        np.testing.assert_array_equal(norms[:,0],norms[:,1])
        np.testing.assert_array_equal(norms[:,:,:,0],norms[:,:,:,1])
        scores={k:z[k] for k in ("native","selected","unselected","whole_site")}
    assert all(v.shape==(256,2,2) and np.isfinite(v).all() for v in scores.values())
    clean=(scores["native"]>0).all(axis=(1,2))
    n=int(clean.sum())
    assert report["eligible_groups"]==n and n>=100
    reversals={k:(v[clean]<0).all(2) for k,v in scores.items() if k!="native"}
    assert report["reversals_by_site"]=={k:v.sum(0).tolist() for k,v in reversals.items()}
    rates={k:float(v.mean()) for k,v in reversals.items()}
    assert report["reversal_rates"]==rates
    effect=(reversals["selected"].astype(int)-reversals["unselected"]).mean(1)
    idx=np.random.default_rng(875024).integers(n,size=(5000,n))
    observed=dict(mean=float(effect.mean()),interval=np.percentile(effect[idx].mean(1),[2.5,97.5]).tolist())
    assert report["selected_minus_unselected"]==observed
    passed=rates["whole_site"]>=.5 and rates["selected"]>=.5 and rates["unselected"]<=.1 and observed["interval"][0]>.25
    assert report["decision"]==("binding_sensitive" if passed else "unresolved")
    print(json.dumps(dict(**report,mean_original_letter_margin={k:v[clean].mean(axis=(0,2)).tolist()
                                                                for k,v in scores.items()}),indent=2))


if __name__=="__main__":
    main()
