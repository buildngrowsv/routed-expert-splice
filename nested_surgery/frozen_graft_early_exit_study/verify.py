"""Audit the early feature prediction and held-out exact complete repairs."""
import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=here/"results"
r=json.loads((root/"result.json").read_text())
receipt=json.loads((root/"receipt.json").read_text())
assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"])==(
    "large-qwen-graft-early-exit","872dbf3e6f9027bb3abd60f2a6eff78413955344","done",0)
assert r["cut"]==10 and r["train_groups"]==192 and r["test_groups"]==512
assert sha256((here.parent/"learned_output_port_study/results/ports.safetensors").read_bytes()).hexdigest()==r["port_sha256"]
prompts=json.loads((root/"programs.json").read_text())
for layout,start,n in (("train",90000,192),("test",91000,512)):
    rows=prompts[layout]
    assert len(rows)==3*n
    assert [v["id"] for v in rows]==[f"opener-{i}-{j}" for i in range(start,start+n) for j in range(3)]
    for i,row in enumerate(rows):
        head,tail=row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        valid=[]
        for k,c in enumerate((")","]","}",":")):
            try:
                ast.parse(head+c+tail)
            except SyntaxError:
                continue
            valid.append(k)
        assert valid==[row["gold"]]==[i%3]
with np.load(root/"maps.npz") as selected:
    assert all(np.isfinite(a).all() for a in selected.values())
    assert selected["mapped_1"].shape==selected["shuffled_1"].shape==(3584,32)
    consistent=selected["mapped_2"]
    np.testing.assert_allclose(consistent,selected["shuffled_2"],rtol=1e-5,atol=1e-6)
with np.load(root/"scores.npz") as z:
    columns=("mapped","shuffled","teacher","host","off","intact")
    assert all(z[k].shape==(512,3,4) and np.isfinite(z[k]).all() for k in columns)
    outcomes={k:(z[k].argmax(-1)==[0,1,2]).all(1) for k in columns}
    signal=z["teacher_features"]
    assert signal.shape==(1536,32)
    errors={k:float(np.mean((z["features_"+k]-signal)**2)) for k in ("mapped","shuffled")}
rawden=np.mean(signal**2)
centered=np.mean((signal-consistent)**2)
for k,v in errors.items():
    np.testing.assert_allclose(v/rawden,r["normalized_feature_errors"][k])
counts={k:int(x.sum()) for k,x in outcomes.items()}
assert counts==r["complete"]
yes=outcomes["mapped"]
strict= yes & outcomes["teacher"] & ~(outcomes["host"]|outcomes["off"]|outcomes["shuffled"])
assert strict.sum()==r["donor_attributable"]
sample=np.random.default_rng(174609).integers(512,size=(5000,512))
for k,s in r["paired"].items():
    d=yes.astype(int)-outcomes[k]
    np.testing.assert_allclose([d.mean(),*np.quantile(d[sample].mean(1),[.025,.975])],[s["mean"],*s["interval"]])
pass_=(counts["teacher"]>=64 and counts["mapped"]>=.8*counts["teacher"] and strict.sum()>=10
       and all(v["mean"]>=.05 and v["interval"][0]>0 for v in r["paired"].values()))
assert r["decision"]==("early_exit_preserves_donor_transfer" if pass_ else "unresolved")
print(json.dumps(dict(counts=counts,attributable=int(strict.sum()),
                      teacher_and_student=int((outcomes["teacher"]&yes).sum()),
                      centered_error={k:v/float(centered) for k,v in errors.items()},
                      paired=r["paired"]),indent=2))
