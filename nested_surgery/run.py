"""A gated few-example transfer from familiar delimiter repairs to nested code."""
from pathlib import Path
import hashlib
import os
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from panels import panel, PATCHES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import prepare
prepare.CACHE = Path(os.environ.get("STUDIO_QWEN_CACHE",Path.home()/"research/routed-expert-splice/cache"))
sys.path.insert(0,str(ROOT/"task_transfer"))
from screen import encode, full_logits, write, load_model
from surgery import train_entrances, modules, ARMS, bf16, create_attention_mask
from delimiters import delimiter_panel, solved
OUT = Path(os.environ["STUDIO_JOB_DIR"])

def interval(a,b):
    delta = a.astype(float)-b.astype(float)
    picks = np.random.default_rng(7812).integers(len(a),size=(5000,len(a)))
    return np.quantile(delta[picks].mean(axis=1),[.025,.975]).tolist()

def final_logits(host, x, pos, head):
    block = host.model.layers[-1]
    mask = create_attention_mask(x,None)
    attended = (x+block.self_attn(block.input_layernorm(x),mask))[mx.arange(len(x)),pos]
    y = host.model.norm(attended+block.mlp(block.post_attention_layernorm(attended)))
    return (y@head.T).astype(mx.float32)

def scores(host, states, positions, head):
    return np.concatenate([np.array(final_logits(host,states[j:j+8],mx.array(positions[j:j+8]),head))
                           for j in range(0,len(states),8)])

def main():
    phase=sys.argv[1]
    host=load_model("host")
    donor=load_model("donor")
    seed={"qualify":1173,"fit":71977,"repeat":93071}[phase]
    rows=panel(seed,256 if phase=="qualify" else 512,(2,3) if phase=="qualify" else (4,6))
    data=encode(rows,PATCHES)
    ids,pos,choices,gold=data
    write(OUT/"programs.json",rows)
    if phase=="qualify":
        logits={k:full_logits(m,data) for k,m in (("donor",donor),("host",host))}
        d,h=(logits[k].argmax(1)==gold for k in ("donor","host"))
        ci=interval(d,h)
        np.savez_compressed(OUT/"logits.npz",gold=gold,**logits)
        advance = d.mean()-h.mean()>=.05 and ci[0]>0
        write(OUT/"result.json",dict(decision="graft" if advance else "stop",donor=int(d.sum()),
            host=int(h.sum()),donor_only=int((d&~h).sum()),host_only=int((h&~d).sum()),n=len(h),interval=ci))
        return
    path=prepare.CACHE/"penultimate/interfaces.safetensors"
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=="e604197879a26d7508491cfd2554b56bd9907460c8cb64e9d35e723909eb06d1"
    arms=modules(host,donor,load_model("general"))
    initial=tree_map(lambda x:x,host.model.layers[-1].parameters())
    head=host.model.embed_tokens.weight[mx.array(choices)]
    pool=delimiter_panel("train")
    rng=np.random.default_rng(seed)
    train_rows=[]
    for k in range(4):
        ix=rng.choice([i for i,r in enumerate(pool) if r["gold"]==k],4,replace=False)
        train_rows.extend([pool[i] for i in ix])
    for row in train_rows:solved(row)
    write(OUT/"training_ids.json",[r["id"] for r in train_rows])
    tr=encode(train_rows,PATCHES)
    train=train_entrances(host,arms,tr[0])
    test=train_entrances(host,arms,ids)
    pre={k:scores(host,test[k],pos,head) for k in ARMS}
    mini=tuple(a[:8] if j!=2 else a for j,a in enumerate(data))
    end_to_end=full_logits(host,mini)
    np.testing.assert_allclose(end_to_end,pre["native"][:8],atol=.25,rtol=.02)
    assert np.array_equal(end_to_end.argmax(1),pre["native"][:8].argmax(1))
    block=host.model.layers[-1]
    logits={"pre-"+k:v for k,v in pre.items()}
    for k in ARMS:
        opt=optim.Adam(2e-5,bias_correction=True)
        weights=tree_map(lambda x:x.astype(mx.float32),initial)
        def objective(w):
            block.update(bf16(w))
            return nn.losses.cross_entropy(final_logits(host,train[k],mx.array(tr[1]),head),
                                            mx.array(tr[-1]),reduction="mean")
        vg=mx.value_and_grad(objective)
        for _ in range(8):
            loss,grad=vg(weights);weights=opt.apply_gradients(grad,weights)
            mx.eval(loss,weights,opt.state)
        block.update(bf16(weights))
        logits[k]=scores(host,test[k],pos,head)
        if k=="coder":
            logits["off"]=scores(host,test["knockout"],pos,head)
            logits["swapped"]=scores(host,test["shuffled"],pos,head)
            block.save_weights(str(OUT/"trained-around-donor.safetensors"))
        block.update(initial)
    ok={k:v.argmax(1)==gold for k,v in logits.items()}
    won=ok["coder"]&~ok["pre-native"]
    strict=won&~ok["native"]
    comparisons={k:dict(delta=float(ok["coder"].mean()-ok[k].mean()),ci=interval(ok["coder"],ok[k])) for k in ("native","shuffled")}
    lost=strict&~ok["off"]
    go=lost.sum()>=10 and all(c["delta"]>=.05 and c["ci"][0]>0 for c in comparisons.values())
    np.savez_compressed(OUT/"logits.npz",gold=gold,**logits)
    write(OUT/"result.json",dict(decision="replicate" if phase=="fit" and go else "complete",seed=seed,n=len(gold),
        solves={k:int(v.sum()) for k,v in ok.items()},comparisons=comparisons,bridge_sha256=digest,
        new_solves=int(won.sum()),new_lost_off=int((won&~ok["off"]).sum()),strict_new=int(strict.sum()),
        strict_lost_ids=[r["id"] for r,b in zip(rows,lost) if b],baseline_lost=int((ok["pre-native"]&~ok["coder"]).sum()),
        off_to_on_net_pre=int(ok["pre-coder"].sum()-ok["pre-knockout"].sum()),
        off_to_on_net_post=int(ok["coder"].sum()-ok["off"].sum())))
if __name__=="__main__":main()

