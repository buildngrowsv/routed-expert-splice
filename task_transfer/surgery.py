"""Train one final host transformer block and score every graft on actual task patches."""
import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from delimiters import delimiter_panel, solved
from screen import ARMS, OUT, WORK, encode, full_logits, load_model, write
from features import WORK as BRIDGES, create_attention_mask, modules, residual_and_input
from forced import SEEDS
from train import CHECKS, bf16


def loss4(logits, y):
    z = logits.astype("float64")
    z -= z.max(axis=1,keepdims=True)
    return np.log(np.exp(z).sum(1)) - z[np.arange(len(y)),y]


def graft_scores(host, graft, dataset):
    layer = host.model.layers[-2]
    original = layer.mlp
    layer.mlp = graft
    try:
        return full_logits(host,dataset)
    finally:
        layer.mlp = original


def train_entrances(host, arms, ids):
    batches = {a:[] for a in ARMS}
    for i in range(0,len(ids),16):
        h,u = residual_and_input(host,mx.array(ids[i:i+16]))
        z = {a:h+f(u) for a,f in arms.items()}
        mx.eval(z)
        for a,x in z.items():
            batches[a].append(x)
    return {a:mx.concatenate(xs) for a,xs in batches.items()}


def fit(host, graft, arm, seed, x_train, train, val, initial, checkpoint_dir=WORK):
    block = host.model.layers[-1]
    master = tree_map(lambda x:x.astype(mx.float32), initial)
    opt = optim.Adam(2e-5,bias_correction=True)
    order = np.random.default_rng(12000+seed).permutation(len(x_train)).reshape(96,16)
    _, pos, choices, y = train
    pos,y = mx.array(pos),mx.array(y)
    head = host.model.embed_tokens.weight[mx.array(choices)]
    def task_loss(w, x, positions, answers):
        block.update(bf16(w))
        mask = create_attention_mask(x,None)
        h = (x+block.self_attn(block.input_layernorm(x),mask))[mx.arange(len(x)),positions]
        output = host.model.norm(h+block.mlp(block.post_attention_layernorm(h)))
        z = (output @ head.T).astype(mx.float32)
        return nn.losses.cross_entropy(z,answers,reduction="mean")
    vg = mx.value_and_grad(task_loss)
    best = float("inf")
    curve = []
    path = checkpoint_dir/f"task-{arm}-{seed}.safetensors"
    for step in range(97):
        if step:
            ix = mx.array(order[step-1].astype("int32"))
            loss,grad = vg(master,x_train[ix],pos[ix],y[ix])
            master = opt.apply_gradients(grad,master)
            mx.eval(master,opt.state,loss)
        if step in CHECKS:
            block.update(bf16(master))
            logits = graft_scores(host,graft,val)
            score = float(loss4(logits,val[-1]).mean())
            curve.append(dict(step=step,loss=score,solves=int((logits.argmax(1)==val[-1]).sum())))
            if score < best:
                best,chosen = score,step
                block.save_weights(str(path))
    block.update(initial)
    return dict(selected=chosen,validation=curve)


def main():
    assert json.loads((OUT/"delimiter-screen.json").read_text())["qualified"]
    host,coder,general = [load_model(n) for n in ("host","donor","general")]
    arms = modules(host,coder,general,fit=False)
    initial = tree_map(lambda x:x,host.model.layers[-1].parameters())
    shapes = {k:v.size for k,v in tree_flatten(initial)}
    path = BRIDGES/"interfaces.safetensors"
    write(OUT/"provenance.json",dict(host_block=27,adapted_block=28,donor_block=24,numbering="one-based",
          params=shapes,trainable=sum(shapes.values()),device=mx.device_info()["device_name"],
          bridges_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),order_seeds=SEEDS))
    train_rows,val_rows = [delimiter_panel(s) for s in ("train","val")]
    for r in train_rows+val_rows:
        solved(r)
    train,val = map(encode,(train_rows,val_rows))
    cache = train_entrances(host,arms,train[0])
    dest = OUT/"selection.json"
    record = json.loads(dest.read_text()) if dest.exists() else {}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{arm}-{seed}"
            if key in record and (WORK/f"task-{arm}-{seed}.safetensors").exists():
                continue
            record[key]=fit(host,arms[arm],arm,seed,cache[arm],train,val,initial)
            write(dest,record)
            print(key,record[key]["selected"],record[key]["validation"][-1]["solves"],flush=True)
    assert len(record)==len(SEEDS)*len(ARMS)


if __name__=="__main__":
    main()
