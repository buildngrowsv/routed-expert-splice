"""Independent replay through the full pretrained model, including same-program exchange."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from experiment import R, SEEDS, checkpoint, corpus, full_logits, load_model, rows_for, sha256, write

class ExchangedOutput(nn.Module):
    def __init__(self, trained, original, positions, shift):
        super().__init__()
        self.trained=trained
        self.original=original
        self.positions=positions
        self.shift=shift
    def __call__(self,x):
        updated=self.trained(x)
        original=self.original(x)
        i=mx.arange(len(x))
        before=original[i,self.positions].astype(mx.float32)
        change=updated[i,self.positions].astype(mx.float32)-before
        permutation=mx.array(np.roll(np.arange(len(x)).reshape(-1,4),self.shift,axis=1).ravel().astype(np.int32))
        updated[i,self.positions]=(before+change[permutation]).astype(updated.dtype)
        return updated


def live_exchange(model,original,data,shift):
    ids,pos,choices,_=data
    layer=model.model.layers[-1]
    trained=layer.mlp
    chunks=[]
    try:
        for i in range(0,len(ids),8):
            at=mx.array(pos[i:i+8])
            layer.mlp=ExchangedOutput(trained,original,at,shift)
            out=model(mx.array(ids[i:i+8]))[mx.arange(len(at)),at]
            chunks.append(np.array(out[:,mx.array(choices)].astype(mx.float32)))
    finally:
        layer.mlp=trained
    return np.concatenate(chunks)


def main():
    selection=json.loads((R/"selection.json").read_text())
    groups={s:set(r["family"] for r in rows_for(s)) for s in ("train","calibration","qualification","test")}
    from indentation.problems import SPLITS
    groups["old_discovery"]={"".join(map(str,f)) for f in SPLITS["discovery"]}
    assert sum(len(g) for g in groups.values())==len(set.union(*groups.values()))
    ledger={}; comparison={}
    for split in ("qualification","test"):
        rows,data=corpus(split)
        assert rows==json.loads((R/f"{split}-problems.json").read_text())
        q=np.arange(len(rows)).reshape(-1,4)
        np.testing.assert_array_equal(data[-1],np.tile(np.arange(4),len(q)))
        for block in q:
            quartet=[rows[j] for j in block]
            assert len({r["prompt"].split("assert tally")[0] for r in quartet})==1
            assert all(quartet[0]["hidden"]==r["hidden"] for r in quartet)
        live={}
        with np.load(R/f"{split}.npz") as saved:
            for name in ("host","donor"):
                model=load_model(name)
                block=model.model.layers[-1]
                original=load_model("donor").model.layers[-1].mlp if name=="donor" else None
                target=block.mlp if name=="donor" else block
                np.testing.assert_array_equal(full_logits(model,data),saved[name+"_base"])
                for seed in SEEDS:
                    path=checkpoint(name,seed)
                    assert sha256(path.read_bytes()).hexdigest()==selection[name][str(seed)]["sha256"]
                    target.load_weights(str(path))
                    z=full_logits(model,data)
                    np.testing.assert_array_equal(z,saved[f"{name}_{seed}"])
                    if name!="donor":continue
                    gold=data[-1]
                    for shift in range(4):
                        key=f"exchange{shift}_{seed}"
                        live[key]=live_exchange(model,original,data,shift)
                        comparison[f"{split}_{key}"]=int((live[key].argmax(1)!=saved[key].argmax(1)).sum())
                        if shift==0: np.testing.assert_array_equal(live[key].argmax(1),z.argmax(1))
                    a=z.argmax(1)==gold
                    b=saved["donor_base"].argmax(1)==gold
                    np.testing.assert_array_equal(saved[f"revert_{seed}"],saved["donor_base"])
                    ledger[f"{split}_{seed}"]=dict(gained_and_lost_on_reversion=[r["id"] for r,keep in zip(rows,a&~b) if keep],
                                                    previously_correct_now_lost=[r["id"] for r,keep in zip(rows,b&~a) if keep])
        np.savez_compressed(R/f"{split}-live-exchange.npz",**live)
    write(R/"reversion-ledger.json",ledger)
    write(R/"replay.json",dict(all_baseline_and_adapted_logits_reproduced=True,all_live_identity_choices_equal_intact=True,
             calibration_and_discovery_families_disjoint=True,all_group_counts={k:len(v) for k,v in groups.items()},
             cached_vs_live_exchange_prediction_disagreements=comparison))
    print("Full-model replay complete",comparison,flush=True)
if __name__=="__main__":main()

