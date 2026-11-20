"""Recover the prespecified endpoint from completed, immutable predictions."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import numpy as np

def paired_interval(a,b):
    diff=a.astype(float)-b.astype(float)
    picks=np.random.default_rng(7812).integers(len(a),size=(5000,len(a)))
    return np.quantile(diff[picks].mean(1),[.025,.975]).tolist()

def summarize(rows,z,gold):
    ok={k:a.argmax(1)==gold for k,a in z.items()};whole={k:v.reshape(-1,3).all(1) for k,v in ok.items()}
    delta=float(whole["coder"].mean()-whole["native"].mean());ci=paired_interval(whole["coder"],whole["native"])
    strict=whole["coder"]&~whole["native"]&~whole["pre-native"]&~whole["off"]
    new=ok["coder"]&~ok["native"]&~ok["pre-native"]&~ok["off"]
    return dict(decision="complete",scores={k:int(v.sum()) for k,v in ok.items()},
        triplets={k:int(v.sum()) for k,v in whole.items()},n_items=len(rows),n_triplets=len(strict),
        strict_disabled_item_ids=[r["id"] for r,v in zip(rows,new) if v],strict_disabled_triplets=np.flatnonzero(strict).tolist(),
        triplet_advantage=delta,interval=ci,supported=bool(delta>=.05 and ci[0]>0 and strict.sum()>=10))

def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs/nested-opener-intervention"
    out=Path(os.environ["STUDIO_JOB_DIR"])
    out.mkdir(parents=True,exist_ok=True)
    digests={p:sha256((root/p).read_bytes()).hexdigest() for p in ("programs.json","logits.npz")}
    assert digests=={"logits.npz":"56df18b66f0c4eca796c3d6db36f474e49b21efcbb80348a2d9c1508541fbb81","programs.json":"1fd9e8dab42e01094f1187666cbca4985447ce40c6a6ab0a0878173168c285b1"}
    rows=json.loads((root/"programs.json").read_text())
    with np.load(root/"logits.npz") as disk:
        z={k:disk[k] for k in disk.files if k!="gold"}
        gold=disk["gold"];np.testing.assert_array_equal(gold,[r["gold"] for r in rows])
        assert len(rows)==768 and all(v.shape==(768,4) and np.isfinite(v).all() for v in z.values())
        report=summarize(rows,z,gold)
    report["recovery_inputs"]=digests
    path=out/"result.json"
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2)+"\n");os.replace(tmp,path)

if __name__=="__main__":main()
