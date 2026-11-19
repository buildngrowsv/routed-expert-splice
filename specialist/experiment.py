"""Train and qualify one coding donor, then exchange only its learned branch contribution."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"task_transfer"))
from screen import CACHE, encode, full_logits, load_model, write
from surgery import fit as host_fit, loss4
from features import create_attention_mask, residual_and_input
from train import CHECKS, bf16
sys.path.insert(0, str(ROOT))
from indentation.problems import PATCHES, SPLITS, panel
from indentation.verify import solve

R = ROOT/"specialist"/"results"
WORK = CACHE/"specialist-v1"
SEEDS = (101,211,307)


def rows_for(split):
    source = "val" if split in ("calibration","qualification") else split
    rows = panel(source, namespace="specialist-v1")
    if source=="val":
        selected = {"".join(map(str,f)) for f in SPLITS["val"][:4]}
        rows = [r for r in rows if (r["family"] in selected)==(split=="calibration")]
    return rows


def corpus(split):
    rows = rows_for(split)
    assert all(solve(r)==r["gold"] for r in rows)
    return rows, encode(rows, PATCHES)


def last_residual(model, ds):
    """Run attention before gathering. The last token-wise MLP can be evaluated after gathering."""
    ids, last, _, _ = ds
    core=model.model
    chunks=[]
    for i in range(0,len(ids),8):
        z=core.embed_tokens(mx.array(ids[i:i+8]))
        mask=create_attention_mask(z,None)
        for layer in core.layers[:-1]:
            z=layer(z,mask)
        block=core.layers[-1]
        h=z+block.self_attn(block.input_layernorm(z),mask)
        h=h[mx.arange(len(h)),mx.array(last[i:i+8])]
        mx.eval(h)
        chunks.append(h)
    return mx.concatenate(chunks)


def donor_fit(model, initial, h, train, cal, seed):
    core=model.model
    block=core.layers[-1]
    mlp=block.mlp
    master=tree_map(lambda a:a.astype(mx.float32),initial)
    opt=optim.Adam(2e-5,bias_correction=True)
    head=core.embed_tokens.weight[mx.array(train[2])]
    labels=mx.array(train[-1])
    order=np.random.default_rng(12000+seed).permutation(len(h)).reshape(96,16)
    def objective(weights,residual,gold):
        mlp.update(bf16(weights))
        out=core.norm(residual+mlp(block.post_attention_layernorm(residual)))
        logits=(out@head.T).astype(mx.float32)
        return nn.losses.cross_entropy(logits,gold,reduction="mean")
    grad=mx.value_and_grad(objective)
    best=float("inf")
    history=[]
    path=WORK/f"task-donor-{seed}.safetensors"
    for step in range(97):
        if step:
            idx=mx.array(order[step-1].astype(np.int32))
            loss,grads=grad(master,h[idx],labels[idx])
            master=opt.apply_gradients(grads,master)
            mx.eval(master,opt.state,loss)
        if step in CHECKS:
            mlp.update(bf16(master))
            logits=full_logits(model,cal)
            score=float(loss4(logits,cal[-1]).mean())
            history.append(dict(step=step,loss=score,solves=int((logits.argmax(1)==cal[-1]).sum())))
            if score<best:
                best=score
                chosen=step
                mlp.save_weights(str(path))
    mlp.update(initial)
    return dict(selected=chosen,validation=history)


def exchange(model,h,old_branch,choices):
    """Keep all recipient features except the learned contribution at the last MLP."""
    block=model.model.layers[-1]
    adapted=block.mlp(block.post_attention_layernorm(h))
    delta=adapted.astype(mx.float32)-old_branch.astype(mx.float32)
    ids=np.arange(len(h)).reshape(-1,4)
    assert ids.shape[0]*4==len(h)
    head=model.model.embed_tokens.weight[mx.array(choices)]
    scores={}
    for shift in range(4):
        take=mx.array(np.roll(ids,shift,axis=1).ravel().astype(np.int32))
        branch=(old_branch.astype(mx.float32)+delta[take]).astype(h.dtype)
        state=model.model.norm(h+branch)
        scores[f"exchange{shift}"]=np.array((state@head.T).astype(mx.float32))
    return scores


def train_arm(name,train,cal):
    model=load_model(name)
    mlp=model.model.layers[-1].mlp
    block=model.model.layers[-1]
    target=mlp if name=="donor" else block
    initial=tree_map(lambda a:a,target.parameters())
    if name=="donor":
        cached=last_residual(model,train)
    else:
        batches=[]
        for i in range(0,len(train[0]),16):
            h,u=residual_and_input(model,mx.array(train[0][i:i+16]))
            x=h+model.model.layers[-2].mlp(u)
            mx.eval(x)
            batches.append(x)
        cached=mx.concatenate(batches)
    mx.eval(cached)
    results={}
    for seed in SEEDS:
        target.update(initial)
        start=time.monotonic()
        mx.reset_peak_memory()
        if name=="donor":
            selection=donor_fit(model,initial,cached,train,cal,seed)
        else:
            selection=host_fit(model,model.model.layers[-2].mlp,"native",seed,cached,train,cal,initial,WORK)
        path=checkpoint(name,seed)
        selection.update(seconds=time.monotonic()-start,peak_mlx_bytes=int(mx.get_peak_memory()),
                         parameters=sum(v.size for _,v in tree_flatten(initial)),sha256=sha256(path.read_bytes()).hexdigest())
        results[str(seed)]=selection
        print(name,seed,selection["selected"],selection["validation"][-1],flush=True)
    return results


def checkpoint(name,seed):
    arm="native" if name=="host" else "donor"
    return WORK/f"task-{arm}-{seed}.safetensors"


def measure(split):
    rows,data=corpus(split)
    results={"gold":data[-1]}
    for name in ("host","donor"):
        model=load_model(name)
        block=model.model.layers[-1]
        target=block.mlp if name=="donor" else block
        initial=tree_map(lambda a:a,target.parameters())
        results[name+"_base"]=full_logits(model,data)
        if name=="donor":
            h=last_residual(model,data)
            old=block.mlp(block.post_attention_layernorm(h))
            mx.eval(old)
        for seed in SEEDS:
            target.load_weights(str(checkpoint(name,seed)))
            results[f"{name}_{seed}"]=full_logits(model,data)
            if name=="donor":
                for k,v in exchange(model,h,old,data[2]).items():
                    results[f"{k}_{seed}"]=v
                target.update(initial)
                results[f"revert_{seed}"]=full_logits(model,data)
                np.testing.assert_array_equal(results[name+"_base"],results[f"revert_{seed}"])
        target.update(initial)
    np.savez_compressed(R/f"{split}.npz",**results)
    (R/f"{split}-problems.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    return rows,results


def interval(v,groups):
    names=np.unique(groups)
    totals=np.array([v[groups==k].sum() for k in names])
    denoms=np.array([(groups==k).sum() for k in names])
    idx=np.random.default_rng(49175).integers(len(names),size=(5000,len(names)))
    return [float(v.mean()),*np.quantile(totals[idx].sum(1)/denoms[idx].sum(1),[.025,.975]).tolist()]


def analyze(rows,predictions):
    gold=predictions["gold"]
    groups=np.array([r["family"] for r in rows])
    out={}
    for name,scores in predictions.items():
        if name=="gold":continue
        hit=scores.argmax(1)==gold
        out[name]=dict(correct=int(hit.sum()),complete_quartets=int(hit.reshape(-1,4).all(1).sum()))
    groupsummary={}
    for seed in SEEDS:
        def hits(name):return predictions[name].argmax(1)==gold
        donor=hits(f"donor_{seed}")
        host=hits(f"host_{seed}")
        raw=hits("donor_base")
        lead=interval(donor.astype(int)-host,groups)
        gain=interval(donor.astype(int)-raw,groups)
        identity=hits(f"exchange0_{seed}")
        scores=np.stack([hits(f"exchange{s}_{seed}") for s in (1,2,3)])
        new=donor&~raw
        groupsummary[str(seed)]=dict(donor_minus_host=lead,donor_minus_base=gain,
              new_lost_on_reversion=int(new.sum()),baseline_solves_lost=int((raw&~donor).sum()),
              identity_prediction_disagreements=int((predictions[f"exchange0_{seed}"].argmax(1)!=predictions[f"donor_{seed}"].argmax(1)).sum()),
              intact_minus_exchange=interval(identity.astype(float)-scores.mean(0),groups),
              intact_donor_only=int((donor&~host).sum()),adapted_host_only=int((host&~donor).sum()),
              gate=bool(lead[0]>=.05 and lead[1]>0 and gain[0]>=.05 and new.any() and host.mean()<.9))
    return dict(n=len(rows),families=sorted(set(groups.tolist())),scores=out,paired=groupsummary)


def main():
    R.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    start=time.monotonic()
    _,train=corpus("train")
    _,cal=corpus("calibration")
    selection={}
    for name in ("donor","host"):
        selection[name]=train_arm(name,train,cal)
        write(R/"selection.json",selection)
    qr,qs=measure("qualification")
    qualification=analyze(qr,qs)
    proceed=all(v["gate"] for v in qualification["paired"].values())
    write(R/"decision.json",dict(qualified=proceed,qualification=qualification))
    print("Gate",proceed,qualification["paired"],flush=True)
    fr,fs=measure("test")
    report=analyze(fr,fs)
    write(R/"test.json",report)
    write(R/"runtime.json",dict(seconds=time.monotonic()-start,device=mx.device_info()["device_name"],stage="specialist qualification"))
    print("Test",report["scores"],flush=True)

if __name__=="__main__": main()

