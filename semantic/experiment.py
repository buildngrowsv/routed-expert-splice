"""Executable four-way repairs, paired counterfactuals, and a gated donor experiment."""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"task_transfer"))
from screen import CACHE, encode, full_logits, load_model, write
from surgery import ARMS, fit, graft_scores, modules, train_entrances, tree_map

R=HERE/"results"
WORK=CACHE/"semantic"
NAMES=tuple("abcd")
SEEDS=(101,211,307)


def panel(split, pairs):
    rows=[]
    for i in range(pairs):
        rng=random.Random(int.from_bytes(hashlib.sha256(f"binding-{split}-{i}".encode()).digest()[:8],"big"))
        perm=rng.sample(range(4),4)
        names=list(NAMES)
        head="def select(values):\n"
        for name,j in zip(names,perm):
            head+=f"    {name} = values[{j}]\n"
        for _ in range(rng.randrange(4)):
            j,k=rng.sample(range(4),2)
            head+=f"    {names[j]}, {names[k]} = {names[k]}, {names[j]}\n"
            perm[j],perm[k]=perm[k],perm[j]
        offset=rng.randint(-99,99)
        head+="    return "
        tail=f" + ({offset})\n"
        cases=[rng.sample(range(-999,1000),4) for _ in range(13)]
        programs=[]
        for n in NAMES:
            env={}
            exec(compile(head+n+tail,"<generated select>","exec"),env)
            programs.append([env["select"](x) for x in cases])
        for side,field in enumerate(rng.sample(range(4),2)):
            answers=[xs[field]+offset for xs in cases]
            passes=[all(a==b for a,b in zip(v[1:],answers[1:])) for v in programs]
            assert sum(passes)==1 and passes[perm.index(field)]
            suffix=tail+f"\n# Contract: select(values) returns values[{field}] + ({offset}).\nassert select({cases[0]}) == {answers[0]}\n"
            prompt="<|fim_prefix|>"+head+"<|fim_suffix|>"+suffix+"<|fim_middle|>"
            rows.append(dict(id=f"{split}-{i:04}-{side}",family="bindings",prompt=prompt,gold=passes.index(True),
                             hidden=cases[1:],expected=answers[1:],passes=passes))
    return rows


def paired_ci(a,b):
    x=(a.astype(float)-b.astype(float)).reshape(-1,2).mean(1)
    idx=np.random.default_rng(5817).integers(len(x),size=(5000,len(x)))
    return [float(x.mean()),*np.quantile(x[idx].mean(1),[.025,.975]).tolist()]


def main():
    WORK.mkdir(exist_ok=True)
    R.mkdir(exist_ok=True)
    host,coder,general=[load_model(n) for n in ("host","donor","general")]
    arms=modules(host,coder,general)
    initial=tree_map(lambda w:w,host.model.layers[-1].parameters())
    tables={}
    def observe(split,n):
        rows=panel(split,n)
        ids=encode(rows,NAMES)
        log={k:full_logits(m,ids) for k,m in (("host",host),("donor",coder))}
        tables[split]=(rows,ids,log)
        return ids,log
    gates={}
    for s,n in (("discovery",128),("val",192)):
        ds,log=observe(s,n)
        h,c=(log[k].argmax(1)==ds[-1] for k in ("host","donor"))
        gates[s]=dict(host=int(h.sum()),donor=int(c.sum()),donor_lead=paired_ci(c,h),n=len(h))
    train=encode(panel("train",768),NAMES)
    val=tables["val"][1]
    from features import residual_and_input
    import mlx.core as mx
    xs=[]
    for i in range(0,len(train[0]),16):
        h,u=residual_and_input(host,mx.array(train[0][i:i+16]))
        x=h+arms["native"](u)
        mx.eval(x)
        xs.append(x)
    native_cache=mx.concatenate(xs)
    fits={"native-101":fit(host,arms["native"],"native",101,native_cache,train,val,initial,WORK)}
    block=host.model.layers[-1]
    block.load_weights(str(WORK/"task-native-101.safetensors"))
    vnative=graft_scores(host,arms["native"],val)
    block.update(initial)
    v=tables["val"][2]
    trained=vnative.argmax(1)==val[-1]
    c=v["donor"].argmax(1)==val[-1]
    h=v["host"].argmax(1)==val[-1]
    go=bool(gates["discovery"]["donor_lead"][0]>=.05 and gates["val"]["donor_lead"][1]>0 and c.mean()-trained.mean()>=.05)
    oracle_room=float((c|h).mean()-max(c.mean(),h.mean()))
    route=oracle_room>=.05
    order=np.sort(v["host"],axis=1)
    margins=order[:,-1]-order[:,-2]
    threshold=-1.
    if route:
        threshold=max([-1.,*sorted(set(map(float,margins)))],
                      key=lambda t:(int(np.where(margins<=t,c,h).sum()),-int((margins<=t).sum())))
    gates.update(native_validation=int(trained.sum()),surgery_qualified=go,oracle_headroom=oracle_room,
                 router_qualified=bool(route),router_threshold=threshold)
    write(R/"gates.json",gates)
    print("Decisions",gates,flush=True)
    if go:
        cache=train_entrances(host,arms,train[0])
        for seed in SEEDS:
            for a in ARMS:
                if a=="native" and seed==101: continue
                fits[f"{a}-{seed}"]=fit(host,arms[a],a,seed,cache[a],train,val,initial,WORK)
    write(R/"selection.json",fits)
    test,log=observe("test",256)
    if route:
        sorted_scores=np.sort(log["host"],axis=1)
        mask=sorted_scores[:,-1]-sorted_scores[:,-2]<=threshold
        log["routed"]=np.where(mask[:,None],log["donor"],log["host"])
    else:
        mask=np.zeros(len(test[0]),dtype=bool)
    for key in fits:
        a,seed=key.split("-")
        block.load_weights(str(WORK/f"task-{a}-{seed}.safetensors"))
        log[key]=graft_scores(host,arms[a],test)
        if a=="coder":
            log["off-"+seed]=graft_scores(host,arms["knockout"],test)
    block.update(initial)
    gold=test[-1]
    ok={k:l.argmax(1)==gold for k,l in log.items()}
    report={k:dict(solves=int(z.sum()),pairs=int(z.reshape(-1,2).all(1).sum())) for k,z in ok.items()}
    report["oracle"]=int((ok["host"]|ok["donor"]).sum())
    report["donor_minus_host"]=paired_ci(ok["donor"],ok["host"])
    report["routed_donor_calls"]=int(mask.sum())
    if route:
        report["route_minus_host"]=paired_ci(ok["routed"],ok["host"])
        report["route_minus_donor"]=paired_ci(ok["routed"],ok["donor"])
    write(R/"test.json",report)
    write(R/"test-problems.json",tables["test"][0])
    for split,(_,data,ls) in tables.items():
        np.savez_compressed(R/(split+"-logits.npz"),gold=data[-1],**ls)
    print("Test",report,flush=True)


if __name__=="__main__":
    main()
