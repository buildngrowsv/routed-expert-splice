"""Audit matched lookup states and exact intervention rates at block 40."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import numpy as np

here=Path(__file__).resolve().parent
root=here/"results"
prior=here.parent/"numeric_lookup_alias_depth_72b_study/results/depth.npz"
panel=here.parent/"numeric_lookup_alias_72b_study/results/programs.json"
receipt=json.loads((root/"receipt.json").read_text())
assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"])==(
    "qwen-dictionary-alias-use-72b","cd94356923c246be5bcbe549bfb32dd18d37cd88","done",0)
r=json.loads((root/"result.json").read_text())
assert r["cut"]==40 and sha256(prior.read_bytes()).hexdigest()==r["compared_ranks_sha256"]
assert sha256(panel.read_bytes()).hexdigest()==r["panel_sha256"]
helper=runpy.run_path(str(here.parent/"numeric_lookup_72b_study/verify.py"))
rows=json.loads(panel.read_text())
with np.load(root/"alias-use.npz") as x,np.load(prior) as y:
    eligible=helper["outcome"](y["native_alias"]) & helper["outcome"](y["native_relocated"])
    assert eligible.sum()==r["shared"]==143
    arm={"alias_copy":y["alias_40"],"relocated_definition":y["relocated_40"]}
    arm.update({k:x[k] for k in ["lookup_read","close_bracket","joint"]})
    loc=x["locations"]
    assert loc.shape==(768,2) and (loc[:,1]==loc[:,0]+1).all()
    assert (loc[:,0]>[v["alias_site"] for v in rows]).all()
    assert (loc[:,1]<[len(v["ids"][0]) for v in rows]).all()
    tokens={k:x for k,x in arm.items() if k!="relocated_definition"}
    for ranks in tokens.values():
        helper["RANKS"](rows,ranks)
counts={}
individual={}
switched={}
for label in arm:
    values=arm[label].reshape(256,3,2,4)[eligible]
    direct=values[:,:2,:,1]==1
    switched[label]=direct.all(axis=(1,2))
    counts[label]={"switch":int(switched[label].sum()),"preserve":int((values[:,2,:,0]==1).all(1).sum())}
    individual[label]=int(direct.sum())
assert counts==r["counts"]
draws=np.random.default_rng(741195).integers(143,size=(8000,143))
for source,comp in r["paired"].items():
    for target,check in comp.items():
        delta=switched[source].astype(int)-switched[target]
        bounds=np.quantile(delta[draws].mean(1),[.0125,.9875])
        np.testing.assert_allclose([delta.mean(),*bounds],[check["mean"],*check["interval"]])
assert all(counts[k]["switch"]==0 for k in ("lookup_read","close_bracket","joint","alias_copy"))
assert r["decision"]=="unresolved"
print(json.dumps(dict(counts=counts,individual_switches=individual,individual_total=572,group_total=143),indent=2))
