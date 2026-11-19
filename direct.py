"""Cheap replacement check. Fit a single affine host-to-host map on training states."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from prepare import CACHE, POS, ridge, load_model, write
from forced import RES, states
from run import bootstrap_files


def main():
    mx.set_default_device(mx.gpu)
    host=load_model("host");core=host.model
    with np.load(CACHE/"train-tokens.npz") as f:train=f["tokens"]
    xs=[];ys=[]
    for i in range(0,len(train),8):
        _,u,v,_=states(host,mx.array(train[i:i+8,:64]))
        for dest,a in ((xs,u),(ys,v)):
            dest.append(np.array(a[:,mx.array(POS)].astype(mx.float32)).reshape(-1,1536))
    w,b=map(mx.array,ridge(np.concatenate(xs),np.concatenate(ys)))
    with np.load(CACHE/"fresh-tokens.npz") as f:
        test=f["tokens"];domain=np.repeat(f["domain"],4);files=np.repeat(f["path"],4)
    class Direct(nn.Module):
        def __call__(self,u):
            return (u.astype(mx.float32)@w+b).astype(mx.bfloat16)
    last=core.layers[-1];old=last.mlp;losses=[];hits=[]
    try:
        last.mlp=Direct()
        for i in range(0,len(test),8):
            ids=mx.array(test[i:i+8,:64]);labels=mx.array(test[i:i+8,POS+1])
            logits=host(ids)[:,mx.array(POS)].astype(mx.float32)
            losses.append(np.array(nn.losses.cross_entropy(logits,labels,reduction="none")).ravel())
            hits.append(np.array(logits.argmax(-1)==labels).ravel())
    finally:last.mlp=old
    l=np.concatenate(losses);h=np.concatenate(hits)
    res={};contrast={}
    for k,n in enumerate(("code","docs")):
        ix=domain==k;res[n]=dict(nll=float(l[ix].mean()),accuracy=float(h[ix].mean()))
        for arm in ("intact","intact-fit","native"):
            with np.load(RES/(arm+".npz")) as f: diff=f["nll"][:,ix].astype("float64")-l[ix]
            contrast.setdefault(arm,{})[n]=dict(mean=float(diff.mean()),interval=bootstrap_files(diff,files[ix]))
    np.savez_compressed(RES/"direct-test.npz",nll=l,correct=h)
    write(RES/"direct.json",{"score":res,"other_minus_direct":contrast,"fitted_values":int(w.size+b.size)})
    print(res,contrast,flush=True)


if __name__=="__main__":main()
