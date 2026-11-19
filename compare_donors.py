"""Matched comparison of the final feed-forward blocks of two real 0.5B donors."""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from prepare import CACHE, OUT, POS, ridge, load_model, write
from forced import states, Connector, Replacement, new_test
from run import bootstrap_files

GENERAL="060db6499f32faf8b98477b0a26969ef7d8b9987"


def main():
    out=OUT/"donors";out.mkdir(parents=True,exist_ok=True)
    snapshot_download("Qwen/Qwen2.5-0.5B",revision=GENERAL,local_dir=CACHE/"general",
                      allow_patterns=["model.safetensors","config.json","tokenizer.json","tokenizer_config.json"])
    t1=json.loads((CACHE/"donor/tokenizer.json").read_text())
    t2=json.loads((CACHE/"general/tokenizer.json").read_text())
    assert all(t1[k]==t2[k] for k in ("model","normalizer","pre_tokenizer"))
    new_test(previous=OUT/"forced/windows.json",seed=15721,name="donors",output=out/"windows.json")
    with np.load(CACHE/"train-tokens.npz") as z:train=z["tokens"]
    with np.load(CACHE/"donors-tokens.npz") as z:
        test=z["tokens"];domain=np.repeat(z["domain"],4);files=np.repeat(z["path"],4)
    host=load_model("host");orig=host.model.layers[-1].mlp
    def get_states(net,indices):
        chunks=[[] for _ in indices]
        for i in range(0,len(train),8):
            observed=states(net,mx.array(train[i:i+8,:64]))
            for j,part in enumerate(indices):
                x=observed[part][:,mx.array(POS)]
                chunks[j].append(np.array(x.astype(mx.float32)).reshape(-1,x.shape[-1]))
        return [np.concatenate(c) for c in chunks]
    xu,yh=get_states(host,(1,2));losses={};report={}
    for arm in ("native","general","donor"):
        if arm!="native":
            small=load_model(arm)
            (yd,)=get_states(small,(1,));wi,bi=map(mx.array,ridge(xu,yd))
            rep=Replacement(small.model.layers[-1].mlp,"intact",wi,bi,None,None,Connector(0,1536))
            features=[]
            for i in range(0,len(xu),128):
                features.append(np.array(rep.expert_output(mx.array(xu[i:i+128])).astype(mx.float32)))
            rep.wout,rep.bout=map(mx.array,ridge(np.concatenate(features),yh))
            host.model.layers[-1].mlp=rep
        losses_= [];hits=[]
        try:
            for i in range(0,len(test),8):
                x=mx.array(test[i:i+8,:64]);y=mx.array(test[i:i+8,POS+1])
                z=host(x)[:,mx.array(POS)].astype(mx.float32)
                losses_.append(np.array(nn.losses.cross_entropy(z,y,reduction="none")).ravel())
                hits.append(np.array(z.argmax(-1)==y).ravel())
        finally:host.model.layers[-1].mlp=orig
        l=np.concatenate(losses_);h=np.concatenate(hits);losses[arm]=l
        report[arm]={n:dict(nll=float(l[domain==k].mean()),accuracy=float(h[domain==k].mean())) for k,n in enumerate(("code","docs"))}
        np.savez_compressed(out/(arm+".npz"),nll=l,correct=h)
    diff={}
    for k,n in enumerate(("code","docs")):
        ix=domain==k;d=losses["donor"][ix].astype("float64")-losses["general"][ix]
        diff[n]=dict(mean=float(d.mean()),interval=bootstrap_files(d[None,:],files[ix]))
    np.savez_compressed(out/"test-groups.npz",domain=domain,files=files)
    write(out/"summary.json",dict(results=report,specialist_minus_general=diff,general_revision=GENERAL))
    print(report,diff,flush=True)


if __name__=="__main__":main()
