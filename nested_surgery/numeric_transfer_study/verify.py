"""Check source independence, executable answers, doses and cross-model reversals."""
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent/"results"
STEM = re.compile(r"amount_\d+")


def check_row(row):
    ids = row["ids"]
    assert [i for i,(a,b) in enumerate(zip(*ids,strict=True)) if a!=b] == [row["site"]]
    for expected,prompt in enumerate(row["prompts"]):
        for j,answer in enumerate(row["answers"]):
            env = {}
            try:
                exec(compile(prompt+answer+"\n","<archived operand test>","exec"),env)
            except AssertionError:
                assert j!=expected
            else:
                value = row["source_values"][expected]
                assert j==expected and env["fetch"]()==int(answer)==value+row["offset"]


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["exit"]==0
    assert receipt["revision"]=="3eacf5424a46a4e2efb328de5fb0ad77f391bc15"
    report = json.loads((ROOT/"result.json").read_text())
    prompts = json.loads((ROOT/"programs.json").read_text())
    assert [r["id"] for r in prompts["donor"]]==list(range(70000,70512))
    assert [r["id"] for r in prompts["host"]]==list(range(62000,62512))
    for donor,host in zip(prompts["donor"],prompts["host"],strict=True):
        assert donor["source_values"]==host["source_values"]
        assert donor["offset"]==0 and host["offset"]!=0
        assert set(map(int,host["answers"])).isdisjoint(donor["source_values"])
        assert STEM.search(donor["prompts"][0]).group() not in "".join(host["prompts"])
        check_row(donor)
        check_row(host)
    with np.load(ROOT/"map.npz") as z:
        assert z["mapping"].shape==(7,7) and np.isfinite(z["mapping"]).all()
        for k,width in (("donor_axis",3584),("host_axis",5120),("random_host_axis",5120)):
            axis = z[k].astype(np.float64)
            assert axis.shape==(width,7)
            np.testing.assert_allclose(axis.T@axis,np.eye(7),atol=1e-6)
        np.testing.assert_allclose(z["host_axis"].astype(np.float64).T@z["random_host_axis"],0,atol=1e-6)
    with np.load(ROOT/"numeric-transfer.npz") as z:
        assert set(z.files)=={"norms","native","transfer","norm_matched","whole_site"}
        doses = z["norms"]
        assert doses.shape==(1024,3) and np.isfinite(doses).all()
        np.testing.assert_allclose(doses[:,0],doses[:,1],atol=1e-6,rtol=1e-5)
        scores = {k:z[k] for k in ("native","transfer","norm_matched","whole_site")}
    assert all(a.shape==(512,2) and np.isfinite(a).all() for a in scores.values())
    eligible = (scores["native"]>0).all(1)
    n = int(eligible.sum())
    assert n==report["eligible"] and n>=100
    flips = {k:(a[eligible]<0).all(1) for k,a in scores.items() if k!="native"}
    counts = {k:int(v.sum()) for k,v in flips.items()}
    assert counts==report["switches"]
    assert report["item_switches"]=={k:(scores[k][eligible]<0).sum(0).tolist() for k in flips}
    d = flips["transfer"].astype(int)-flips["norm_matched"]
    samples = np.random.default_rng(720504).integers(n,size=(4000,n))
    contrast = dict(mean=float(d.mean()),interval=np.percentile(d[samples].mean(1),[2.5,97.5]).tolist())
    assert contrast==report["transfer_minus_random"]
    offsets = np.array([r["offset"] for r in prompts["host"]])
    for op,sign in (("addition",1),("subtraction",-1)):
        selection = eligible & (np.sign(offsets)==sign)
        summary = dict(eligible=int(selection.sum()),
                       reversals={k:int((scores[k][selection]<0).all(1).sum()) for k in flips})
        assert summary==report["by_operator"][op]
    qualified = min(counts[k] for k in ("transfer","whole_site"))>=.8*n and d.mean()>=.2 and contrast["interval"][0]>0
    assert report["decision"]==("transfer" if qualified else "stop")
    audit = dict(**report,mean_margin={k:float(v[eligible].mean()) for k,v in scores.items()},
                 map_sha256=sha256((ROOT/"map.npz").read_bytes()).hexdigest(),
                 max_norm_error=float(np.max(np.abs(doses[:,0]-doses[:,1]))))
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
