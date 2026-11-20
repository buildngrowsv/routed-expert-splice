"""Reconstruct the fixed-input intervention and its paired margins."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=here/"results"
base=here.parent/"frozen_graft_early_exit_study/results"
receipt=json.loads((root/"receipt.json").read_text())
assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"])==(
    "large-qwen-graft-constant-control","131020da906d6f8c546e3aa43e9272c3b264548d","done",0)
r=json.loads((root/"result.json").read_text())
assert r["groups"]==512
assert sha256((base/"programs.json").read_bytes()).hexdigest()==r["panel_sha256"]
assert sha256((here.parent/"learned_output_port_study/results/ports.safetensors").read_bytes()).hexdigest()==r["port_sha256"]
with np.load(root/"constant.npz") as evidence,np.load(base/"scores.npz") as earlier:
    assert set(evidence)=={"aligned","calibration_mean","cyclic"}
    scores={k:earlier[k] for k in ("mapped","host","teacher","shuffled","off")}
    scores.update({k:evidence[k] for k in evidence})
    assert all(v.shape==(512,3,4) and np.isfinite(v).all() for v in scores.values())
    ids={k:v.argmax(-1) for k,v in scores.items()}
np.testing.assert_array_equal(ids["aligned"],ids["mapped"])
success={k:(v==[0,1,2]).all(axis=-1) for k,v in ids.items()}
assert {k:int(v.sum()) for k,v in success.items()}==r["complete"]
correct=success["aligned"]
constant=success["calibration_mean"]
strict=success["teacher"] & correct & ~(constant|success["cyclic"]|success["host"]|success["shuffled"]|success["off"])
assert strict.sum()==r["teacher_shared_strict"]
assert (ids["cyclic"]==[1,2,0]).all(axis=-1).sum()==r["exact_cyclic_follows"]
draws=np.random.default_rng(731817).integers(512,size=(5000,512))
for key,a,b in (("early_minus_constant",correct,constant),("constant_minus_host",constant,success["host"])):
    d=a.astype(int)-b
    np.testing.assert_allclose([d.mean(),*np.quantile(d[draws].mean(1),[.025,.975])],[r[key]["mean"],*r[key]["interval"]])
change,adv=r["early_minus_constant"],r["constant_minus_host"]
decision="unresolved"
if change["mean"]>=.05 and change["interval"][0]>0 and strict.sum()>=10:
    decision="prompt_specific_donor_features_help"
elif change["interval"][1]<=.02 and adv["mean"]>=.05 and adv["interval"][0]>0:
    decision="constant_feature_matches_early_graft"
assert r["decision"]==decision
print(json.dumps(dict(scores=r["complete"],constant_gained=int((constant&~correct).sum()),
                      constant_lost=int((~constant&correct).sum()),decision=decision,
                      early_minus_constant=change),indent=2))
