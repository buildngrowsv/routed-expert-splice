"""Counterfactual opening-delimiter transfers with an identical FIM suffix."""
import ast
from opener_report import summarize
import sys
import json
import random
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nested_surgery.run import (prepare,load_model,encode,modules,tree_map,train_entrances,delimiter_panel,
                 final_logits,bf16,graft_scores,full_logits,PATCHES,OUT,write,interval)
import os

def programs(n):
    rows=[]
    for i in range(n):
        rng=random.Random(f"opener-transfer-v1-{i}")
        expr="x"
        for _ in range(rng.randint(4,6)):
            expr=f"({expr} + ({rng.randrange(-99,100)}))"
            expr=rng.choice(["("+expr+")","sum(("+expr+", 0))","["+expr+", 0][0]"])
        inputs=[rng.randrange(-100,100) for _ in range(12)]
        for k,c in enumerate("([{"):
            head="def transform(x):\n    value = "+c+expr
            tail="\n    return x\n"
            valid=[]
            for j,patch in enumerate(PATCHES):
                try:tree=ast.parse(head+patch+tail)
                except SyntaxError:continue
                env={"__builtins__":{"sum":sum}}
                exec(compile(tree,"<generated opener>","exec"),env)
                assert all(env["transform"](x)==x for x in inputs)
                valid.append(j)
            assert valid==[k]
            rows.append(dict(id=f"opener-{i}-{k}",family="delimiters",gold=k,
                prompt="<|fim_prefix|>"+head+"<|fim_suffix|>"+tail+"<|fim_middle|>"))
        assert sum(a!=b for a,b in zip(rows[-3]["prompt"],rows[-1]["prompt"]))==1
    return rows

def main():
    runs=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"; fit=runs/"nested-donor-full-forward"
    rows=programs(256);data=encode(rows,PATCHES);gold=data[-1]
    write(OUT/"programs.json",rows)
    host=load_model("host");small=load_model("donor")
    arms=modules(host,small,load_model("general"));block=host.model.layers[-1]
    initial=tree_map(lambda w:w,block.parameters())
    ids=json.loads((fit/"training_ids.json").read_text());pool={r["id"]:r for r in delimiter_panel("train")}
    tr=encode([pool[k] for k in ids],PATCHES);x=train_entrances(host,arms,tr[0])
    head=host.model.embed_tokens.weight[mx.array(tr[2])]
    z={"pre-native":full_logits(host,data),"standalone-donor":full_logits(small,data)}
    for name in ("native","general"):
        weights=tree_map(lambda a:a.astype(mx.float32),initial);opt=optim.Adam(2e-5,bias_correction=True)
        def objective(w):
            block.update(bf16(w))
            v=final_logits(host,x[name],mx.array(tr[1]),head)
            return nn.losses.cross_entropy(v,mx.array(tr[-1]),reduction="mean")
        vg=mx.value_and_grad(objective)
        for _ in range(8):
            loss,g=vg(weights);weights=opt.apply_gradients(g,weights);mx.eval(weights,opt.state,loss)
        block.update(bf16(weights));z[name]=graft_scores(host,arms[name],data);block.update(initial)
    block.load_weights(str(fit/"trained-around-donor.safetensors"))
    for k,arm in (("coder","coder"),("off","knockout"),("swapped","shuffled")):z[k]=graft_scores(host,arms[arm],data)
    np.savez_compressed(OUT/"logits.npz",gold=gold,**z)
    write(OUT/"result.json",summarize(rows,z,gold))
if __name__=="__main__":main()
