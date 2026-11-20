"""Re-execute all completions and audit matched perturbations and paired reversals."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent/"results"


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["exit"]==0
    assert receipt["revision"]=="27a2e3bbccfd7eeda6a8acdc6bf46a5ae69ed934"
    result = json.loads((ROOT/"result.json").read_text())
    rows = json.loads((ROOT/"programs.json").read_text())
    assert [(r["id"],r["query"]) for r in rows] == [(i,k) for i in range(24000,24256) for k in range(2)]
    for first,second in zip(rows[::2],rows[1::2],strict=True):
        stop = first["sites"][-1]+1
        assert first["sites"]==second["sites"] and first["choices"]==second["choices"]
        assert all(a[:stop]==b[:stop] for a,b in zip(first["ids"],second["ids"],strict=True))
        expected = []
        for row in (first,second):
            assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==row["sites"]
            for i,prompt in enumerate(row["prompts"]):
                for j,letter in enumerate(row["letters"]):
                    namespace = {}
                    exec(compile(prompt+letter+"\n","<archived generated binding probe>","exec"),namespace)
                    try:
                        value = namespace["fetch"]()
                    except AttributeError:
                        assert i!=j
                    else:
                        assert i==j
                        expected.append(value)
        assert expected[0]==expected[1] and expected[2]==expected[3] and expected[0]!=expected[2]
    with np.load(ROOT/"binding.npz") as data:
        assert set(data.files)=={"axis","perturbation_norms","native","selected","unselected","whole_site"}
        doses = data["perturbation_norms"]
        assert doses.shape==(256,2,2,2) and np.isfinite(doses).all()
        np.testing.assert_array_equal(doses[:,0],doses[:,1])
        axis = data["axis"].astype(np.float64)
        assert axis.shape==(5120,25)
        np.testing.assert_allclose(axis.T@axis,np.eye(25),atol=1e-6)
        scores = {k:data[k] for k in ("native","selected","unselected","whole_site")}
    assert all(v.shape==(256,2,2) and np.isfinite(v).all() for v in scores.values())
    eligible = (scores["native"]>0).all(axis=(1,2))
    n = int(eligible.sum())
    assert result["eligible_groups"]==n and n>=100
    flips = {k:(v[eligible]<0).all(axis=2) for k,v in scores.items() if k!="native"}
    counts = {k:v.sum(axis=0).tolist() for k,v in flips.items()}
    rates = {k:float(v.mean()) for k,v in flips.items()}
    assert result["reversals_by_site"]==counts and result["reversal_rates"]==rates
    diff = (flips["selected"].astype(int)-flips["unselected"]).mean(axis=1)
    draws = np.random.default_rng(875024).integers(n,size=(5000,n))
    contrast = dict(mean=float(diff.mean()),interval=np.quantile(diff[draws].mean(1),[.025,.975]).tolist())
    assert result["selected_minus_unselected"]==contrast
    passed = rates["whole_site"]>=.5 and rates["selected"]>=.5 and rates["unselected"]<=.1 and contrast["interval"][0]>.25
    assert result["decision"]==("binding_sensitive" if passed else "unresolved")
    print(json.dumps(dict(**result,max_role_dose_mismatch=float(np.max(abs(doses[:,0]-doses[:,1])))),indent=2))


if __name__ == "__main__":
    main()
