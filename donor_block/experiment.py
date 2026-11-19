"""Fit the final coding block once per order and recombine its native parameters."""
from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from specialist.experiment import (CACHE, SEEDS, checkpoint, corpus, encode, full_logits, host_fit,
        load_model, mx, residual_and_input, rows_for, tree_flatten, tree_map, write)
from specialist.experiment import interval as family_interval
from specialist.report import paired_interval
from indentation.problems import PATCHES,panel
from indentation.verify import solve
import numpy as np

R=ROOT/"donor_block"/"results"
WORK=CACHE/"donor-block-v1"
ARMS=("full","attention","expert","previous_expert","host")

def new_panel(split):
    families=[f for f in product(range(6),repeat=3) if sum(v>=4 for v in f)==1]
    assert len(families)==96
    random.Random(46281).shuffle(families)
    pool={"qualification":families[:16],"confirmation":families[16:32]}[split]
    return panel(split,"whole-donor-v1",forms_pool=pool,size=512)

def output_key(prompt):
    base=int(re.search(r"    total = (-?\d+)\n",prompt)[1])
    bonus=int(re.search(r"total \+= (\d+)\n",prompt)[1])
    outputs=[int(x)-base for x in re.findall(r"assert tally\([^\n]+?\) == (-?\d+)\n",prompt)]
    assert len(outputs)==4 and all(x%bonus==0 for x in outputs)
    return tuple(sorted(x//bonus for x in outputs))

def fit_shortcut(rows):
    votes=defaultdict(lambda:np.zeros(4,dtype=int))
    for row in rows:
        votes[output_key(row["prompt"])][row["gold"]]+=1
    return np.array(list(votes)),np.stack(list(votes.values()))

def shortcut_scores(rows,table):
    keys,histograms=table
    scores=[]
    for row in rows:
        distance=np.abs(keys-output_key(row["prompt"])).sum(1)
        scores.append(histograms[distance==distance.min()].sum(0))
    return np.stack(scores)

def trained_path(seed):
    return WORK/f"task-whole-{seed}.safetensors"

def fit():
    _,train=corpus("train")
    _,cal=corpus("calibration")
    model=load_model("donor")
    block=model.model.layers[-1]
    initial=tree_map(lambda x:x,block.parameters())
    inputs=[]
    for i in range(0,len(train[0]),16):
        h,u=residual_and_input(model,mx.array(train[0][i:i+16]))
        state=h+model.model.layers[-2].mlp(u)
        mx.eval(state)
        inputs.append(state)
    states=mx.concatenate(inputs)
    mx.eval(states)
    record={}
    for seed in SEEDS:
        block.update(initial)
        mx.reset_peak_memory()
        start=time.monotonic()
        selection=host_fit(model,model.model.layers[-2].mlp,"whole",seed,states,train,cal,initial,WORK)
        selection.update(parameters=sum(v.size for _,v in tree_flatten(initial)),
            seconds=time.monotonic()-start,peak_bytes=int(mx.get_peak_memory()),
            sha256=sha256(trained_path(seed).read_bytes()).hexdigest())
        record[str(seed)]=selection
        print("fit",seed,selection["selected"],selection["validation"][-1],flush=True)
    write(R/"selection.json",record)

def archived_hashes():
    original=json.loads((ROOT/"specialist"/"results"/"selection.json").read_text())
    for name,seeds in original.items():
        for seed,record in seeds.items():
            assert sha256(checkpoint(name,int(seed)).read_bytes()).hexdigest()==record["sha256"]

def measure(rows,table):
    gold=np.array([solve(r) for r in rows])
    data=encode(rows,PATCHES)
    np.testing.assert_array_equal(data[-1],gold)
    result={"gold":gold,"shortcut":shortcut_scores(rows,table)}
    host=load_model("host")
    result["host_base"]=full_logits(host,data)
    for seed in SEEDS:
        host.model.layers[-1].load_weights(str(checkpoint("host",seed)))
        result[f"host_{seed}"]=full_logits(host,data)
    del host
    donor=load_model("donor")
    block=donor.model.layers[-1]
    raw=tree_map(lambda x:x,block.parameters())
    result["donor_base"]=full_logits(donor,data)
    for seed in SEEDS:
        block.load_weights(str(trained_path(seed)))
        trained=tree_map(lambda x:x,block.parameters())
        assert set(trained)=={"mlp","self_attn","input_layernorm","post_attention_layernorm"}
        for name,values in (("full",trained),("attention",{**trained,"mlp":raw["mlp"]}),
                            ("expert",{**raw,"mlp":trained["mlp"]})):
            block.update(values)
            result[f"{name}_{seed}"]=full_logits(donor,data)
        block.update(raw)
        block.mlp.load_weights(str(checkpoint("donor",seed)))
        result[f"previous_expert_{seed}"]=full_logits(donor,data)
        block.update(raw)
    np.testing.assert_array_equal(full_logits(donor,data),result["donor_base"])
    return result

def analyze(rows,z):
    groups=np.array([r["family"] for r in rows])
    gold=z["gold"]
    hit={k:(v.argmax(1)==gold).astype(int) for k,v in z.items() if k!="gold"}
    arms={k:np.stack([hit[f"{k}_{s}"] for s in SEEDS]) for k in ARMS}
    raw=hit["donor_base"]
    comparisons={
        "full_minus_separately_trained_expert":arms["full"]-arms["previous_expert"],
        "full_minus_host":arms["full"]-arms["host"],
        "expert_minus_host":arms["expert"]-arms["host"],
        "full_minus_attention":arms["full"]-arms["attention"],
        "full_minus_expert":arms["full"]-arms["expert"],
        "attention_minus_raw":arms["attention"]-raw,
        "expert_minus_raw":arms["expert"]-raw,
        "interaction":arms["full"]-arms["expert"]-arms["attention"]+raw}
    effects={k:dict(pooled=paired_interval(v,groups),by_order=[family_interval(each,groups) for each in v])
             for k,v in comparisons.items()}
    passed=all(est[0]>=.05 and est[1]>0 for k in ("full_minus_host","expert_minus_host","full_minus_attention")
               for est in effects[k]["by_order"])
    switches={}
    for j,seed in enumerate(SEEDS):
        changes={}
        for k in ("attention","expert","previous_expert","host"):
            diff=arms["full"][j]-arms[k][j]
            changes[k]=dict(full_only=[r["id"] for r,v in zip(rows,diff) if v==1],
                            other_only=[r["id"] for r,v in zip(rows,diff) if v==-1])
        switches[str(seed)]=changes
    scores={k:dict(solved=int(v.sum()),quartets=int(v.reshape(-1,4).all(1).sum())) for k,v in hit.items()}
    return dict(n=len(rows),families=sorted(set(groups.tolist())),scores=scores,effects=effects,passed=passed),switches

def main():
    R.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    archived_hashes()
    start=time.monotonic()
    fit()
    table=fit_shortcut(rows_for("train"))
    qualification=None
    for split in ("qualification","confirmation"):
        if split=="confirmation":
            assert (R/"decision.json").exists() and qualification is not None
        rows=new_panel(split)
        scores=measure(rows,table)
        report,ledger=analyze(rows,scores)
        if split=="qualification":
            qualification=report["passed"]
            write(R/"decision.json",dict(qualified=qualification))
        np.savez_compressed(R/f"{split}.npz",**scores)
        (R/f"{split}-problems.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
        write(R/f"{split}-summary.json",report)
        write(R/f"{split}-switches.json",ledger)
        print(split,"passed",report["passed"],{k:v["solved"] for k,v in report["scores"].items()},flush=True)
    write(R/"runtime.json",dict(device=mx.device_info()["device_name"],seconds=time.monotonic()-start))
if __name__=="__main__":main()

