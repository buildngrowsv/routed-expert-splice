"""Recompute empirical loss/accuracy and check file and token-window separation."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent/"results"
for folder,names in [(R,["host","unmodified","shuffled","intact","intact-off"]),
                      (R/"forced",["native","knockout","intact","shuffled","intact-fit","shuffled-fit","intact-off","shuffled-off"]),
                      (R/"donors",["native","general","donor"])]:
    groups=np.load(folder/"test-groups.npz");domain=groups["domain"];assert len(domain)==4096
    report=json.loads((folder/"summary.json").read_text())["results"]
    for name in names:
        filename=name+"-test.npz" if folder==R else name+".npz"
        with np.load(folder/filename) as pred:
            for k,dom in enumerate(("code","docs")):
                mask=domain==k
                np.testing.assert_allclose([pred["nll"][...,mask].mean(),pred["correct"][...,mask].mean()],
                                           [report[name][dom]["nll"],report[name][dom]["accuracy"]],rtol=1e-6)
with np.load(R/"forced/direct-test.npz") as f,np.load(R/"forced/test-groups.npz") as g:
    d=json.loads((R/"forced/direct.json").read_text())["score"]
    for k,n in enumerate(("code","docs")):
        np.testing.assert_allclose(f["nll"][g["domain"]==k].mean(),d[n]["nll"])
original=json.loads((R/"windows.json").read_text());seen=set();split_files={}
def check_window(path,off,digest):
    assert digest not in seen and (path,off) not in seen and off%65==0
    seen.add(digest);seen.add((path,off))
for split,domain,path,off,digest in original:
    check_window(path,off,digest);split_files.setdefault(split,set()).add(path)
assert not any(split_files[a]&split_files[b] for a,b in (("train","val"),("train","test"),("val","test")))
for label in ("forced","donors"):
    for _,path,off,digest in json.loads((R/label/"windows.json").read_text()):
        assert path in split_files["test"];check_window(path,off,digest)
print("All three panels, the affine control, file splits and non-overlapping held-out windows verified.")
