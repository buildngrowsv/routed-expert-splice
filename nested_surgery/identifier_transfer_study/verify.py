"""Reconstruct paired reversals, verify source map shapes, and check doses."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent/"results"


def main():
    report = json.loads((ROOT/"result.json").read_text())
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["exit"]==0
    assert receipt["revision"]=="27a2e3bbccfd7eeda6a8acdc6bf46a5ae69ed934"
    rows = json.loads((ROOT/"test-programs.json").read_text())
    assert [r["id"] for r in rows]==list(range(16000,16512))
    for row in rows:
        assert [i for i,(a,b) in enumerate(zip(*row["ids"],strict=True)) if a!=b]==[row["site"]]
        for prompt,letter in zip(row["prompts"],row["letters"],strict=True):
            env = {}
            exec(compile(prompt+letter+"\n","<archived generated identifier>","exec"),env)
            assert env["fetch"]()==7
    with np.load(ROOT/"map.npz") as data:
        assert data["mapping"].shape==(25,25) and np.isfinite(data["mapping"]).all()
        for name,width in (("donor_axis",3584),("host_axis",5120),("random_host_axis",5120)):
            b = data[name].astype(np.float64)
            assert b.shape==(width,25)
            np.testing.assert_allclose(b.T@b,np.eye(25),atol=1e-6)
        np.testing.assert_allclose(data["host_axis"].astype(np.float64).T@data["random_host_axis"],0,atol=1e-6)
    with np.load(ROOT/"transfer.npz") as data:
        assert set(data.files)=={"norms","native","transfer","norm_matched","whole_site"}
        shift = data["norms"]
        assert shift.shape==(1024,3) and np.isfinite(shift).all()
        np.testing.assert_allclose(shift[:,0],shift[:,1],atol=1e-6,rtol=1e-5)
        logits = {k:data[k] for k in ("native","transfer","norm_matched","whole_site")}
    assert all(s.shape==(512,2) and np.isfinite(s).all() for s in logits.values())
    clean = (logits["native"]>0).all(1)
    n = int(clean.sum())
    assert n==report["eligible"] and n>=100
    changed = {k:(v[clean]<0).all(1) for k,v in logits.items() if k!="native"}
    counts = {k:int(v.sum()) for k,v in changed.items()}
    assert report["switches"]==counts
    assert report["item_switches"]=={k:(v[clean]<0).sum(0).tolist() for k,v in logits.items() if k!="native"}
    d = changed["transfer"].astype(int)-changed["norm_matched"]
    idx = np.random.default_rng(720504).integers(n,size=(4000,n))
    observed = dict(mean=float(d.mean()),interval=np.quantile(d[idx].mean(1),[.025,.975]).tolist())
    assert observed==report["transfer_minus_random"]
    passed = counts["whole_site"]>=.8*n and counts["transfer"]>=.8*n and observed["mean"]>=.2 and observed["interval"][0]>0
    assert report["decision"]==("transfer" if passed else "stop")
    out = dict(**report,mean_margin={k:float(v.mean()) for k,v in logits.items()},
               max_norm_difference=float(abs(shift[:,0]-shift[:,1]).max()),
               median_transfer_to_full_norm=float(np.median(shift[:,0]/shift[:,2])),
               map_sha256=sha256((ROOT/"map.npz").read_bytes()).hexdigest())
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
