"""Recompute reversal counts and match the dose from both score archives."""
import ast
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE/"results"


def main():
    report = json.loads((ROOT/"result.json").read_text())
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="9744af55c991b08c1db5a0880d099513c078c242"
    rows = json.loads((ROOT/"test-programs.json").read_text())
    assert [r["id"] for r in rows]==list(range(8192,8704))
    for row in rows:
        assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==[row["site"]]
        for p,x in zip(row["prompts"],row["letters"],strict=True):
            ast.parse(p+x+"\n")
    out = {}
    for key,depth in (("donor",6),("host",10)):
        with np.load(ROOT/(key+"-margins.npz")) as archive:
            data = dict(archive)
        dose = data.pop("norms")
        assert dose.shape==(1024,3) and np.isfinite(dose).all() and np.all(dose>0)
        np.testing.assert_allclose(dose[:,1],dose[:,2],rtol=1e-5,atol=1e-6)
        assert all(v.shape==(512,2) and np.isfinite(v).all() for v in data.values())
        good = (data["native"]>0).all(1)
        n = int(good.sum())
        changed = {k:(v[good]<0).all(1) for k,v in data.items() if k!="native"}
        n_changed = {k:int(v.sum()) for k,v in changed.items()}
        diff = changed["learned"].astype(int)-changed["matched"]
        ix = np.random.default_rng(50324).choice(n,(5000,n))
        ci = np.quantile(diff[ix].mean(1),[.025,.975])
        estimate = dict(mean=float(diff.mean()),interval=ci.tolist())
        passed = (n>=100 and n_changed["whole_site"]>=.2*n and n_changed["learned"]>=.2*n
                  and diff.mean()>=.10 and ci[0]>0)
        assert report["arms"][key]==dict(eligible=n,switches=n_changed,rank=25,depth=depth,qualifies=bool(passed),
                                         learned_minus_matched=estimate)
        out[key] = {**report["arms"][key],"median_projected_norm_fraction":float(np.median(dose[:,1]/dose[:,0])),
                    "max_dose_mismatch":float(abs(dose[:,1]-dose[:,2]).max()),
                    "average_correct_log_odds":{k:float(v.mean()) for k,v in data.items()}}
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
