"""Train host adapters and score the locked surgery protocol on the Studio GPU."""
from pathlib import Path
import time
import json
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from prepare import CACHE, OUT, POS, write

ARMS=["host","shuffled","intact"]
RANK=16


class Surgery(nn.Module):
    def __init__(self,arm,seed,dim):
        super().__init__()
        rng=np.random.default_rng(seed)
        def proj():return mx.array(rng.normal(0,dim**-.5,(dim,RANK)).astype("float32"))
        self.arm=arm
        self.host_in=proj();self.host_out=mx.zeros((RANK,dim))
        if arm!="host":
            self.donor_in=proj();self.donor_out=mx.zeros((RANK,dim))
            self.router=mx.zeros((dim,1));self.bias=mx.array([-2.94])

    def __call__(self,u,base,native,delta,off=False):
        x=mx.fast.rms_norm(u.astype(mx.float32),None,1e-5)
        h=base.astype(mx.float32)+(x@self.host_in)@self.host_out
        if self.arm=="host" or off:
            return h,mx.zeros((x.shape[0],1))
        gate=mx.sigmoid(x@self.router+self.bias)
        donor=native.astype(mx.float32)+delta
        y=mx.fast.rms_norm(donor,None,1e-5)
        return h+gate*(delta+(y@self.donor_in)@self.donor_out),gate


def load(split):
    with np.load(CACHE/("host-"+split+".npz")) as f: states={k:mx.array(f[k]) for k in f.files}
    with np.load(CACHE/("branches-"+split+".npz")) as f:states.update({k:mx.array(f[k]) for k in f.files})
    with np.load(CACHE/(split+"-tokens.npz")) as f:
        return states,mx.array(f["tokens"][:,POS+1].reshape(-1)),np.repeat(f["domain"],len(POS)),np.repeat(f["path"],len(POS))


def bootstrap_files(diff,paths):
    files,idx=np.unique(paths,return_inverse=True)
    lengths=np.bincount(idx)
    sums=np.stack([np.bincount(idx,weights=row) for row in diff])
    rng=np.random.default_rng(92491);n=5000
    w=rng.multinomial(len(files),np.full(len(files),1/len(files)),n)
    means=(w@sums.T)/(w@lengths)[:,None]
    seed_draw=rng.integers(diff.shape[0],size=(n,diff.shape[0]))
    return np.quantile(np.take_along_axis(means,seed_draw,axis=1).mean(axis=1),[.025,.975]).tolist()


def main():
    mx.set_default_device(mx.gpu);OUT.mkdir(exist_ok=True)
    raw=mx.load(str(CACHE/"host/model.safetensors"))
    head=raw["model.embed_tokens.weight"];norm=raw["model.norm.weight"]
    mx.eval(head,norm);del raw
    def logits(h):
        y=mx.fast.rms_norm(h.astype(mx.bfloat16),norm,1e-6)
        return (y@head.T).astype(mx.float32)
    def losses(m,u,base,ff,diff,y):
        h,_=m(u,base,ff,diff);return nn.losses.cross_entropy(logits(h),y,reduction="mean")
    def eval_model(net,data,off=False):
        st,y,domain,files=data
        out=[];correct=[];gates=[]
        delta=st["intact" if net.arm=="host" else net.arm]
        for i in range(0,len(domain),96):
            sl=slice(i,i+96)
            h,g=net(st["u"][sl],st["base"][sl],st["native"][sl],delta[sl],off)
            log=logits(h);loss=nn.losses.cross_entropy(log,y[sl],reduction="none")
            out.append(np.array(loss));correct.append(np.array(log.argmax(1)==y[sl]));gates.append(np.array(g)[:,0])
        loss,hit,gate=map(np.concatenate,(out,correct,gates))
        metrics={name:dict(nll=float(loss[domain==k].mean()),accuracy=float(hit[domain==k].mean()),
                           gate=float(gate[domain==k].mean())) for k,name in enumerate(("code","docs"))}
        return metrics,loss,hit,gate
    train,val=load("train"),load("val")
    def fit(arm,seed):
        net=Surgery(arm,seed,head.shape[1]);opt=optim.Adam(.001)
        state=[net.state,opt.state];vg=nn.value_and_grad(net,losses)
        @partial(mx.compile,inputs=state,outputs=state)
        def step(u,base,ff,diff,y):
            loss,g=vg(net,u,base,ff,diff,y);opt.update(net,g);return loss
        st,y,domain,_=train
        delta=st["intact" if arm=="host" else arm]
        rng=np.random.default_rng(seed+12000);ck=CACHE/f"best-{seed}-{arm}.safetensors"
        curve=[];best=float("inf");epoch=None;start=time.perf_counter()
        for ep in range(5):
            if ep:
                order=mx.array(rng.permutation(len(domain)).astype("int32"))
                for j in range(0,len(domain),64):
                    ix=order[j:j+64]
                    loss=step(st["u"][ix],st["base"][ix],st["native"][ix],delta[ix],y[ix])
                    mx.eval(loss,state)
            m,*_=eval_model(net,val);curve.append(m)
            score=(m["code"]["nll"]+m["docs"]["nll"])/2
            if score<best:
                best=score;epoch=ep;net.save_weights(str(ck))
        res=dict(seed=seed,arm=arm,epoch=epoch,validation=curve[epoch],curve=curve,
                 trainable=sum(v.size for _,v in tree_flatten(net.trainable_parameters())),
                 training_and_selection_seconds=time.perf_counter()-start)
        write(OUT/f"{seed}-{arm}.json",res)
        print("selected",seed,arm,epoch,res["validation"],flush=True)
        mx.clear_cache()
    seeds=[101,211,307]
    def runseeds(ss):
        for s in ss:
            for arm in np.random.default_rng(s).permutation(ARMS):
                if not (OUT/f"{s}-{arm}.json").exists():fit(arm,s)
    runseeds(seeds)
    means={}
    for arm in ARMS:
        selected=[json.loads((OUT/f"{s}-{arm}.json").read_text())["validation"] for s in seeds]
        means[arm]={k:float(np.mean([m[k]["nll"] for m in selected])) for k in ("code","docs")}
    expand=(all(means["intact"]["code"]<=means[a]["code"]-.02 for a in ARMS[:-1])
            and means["intact"]["docs"]<=means["host"]["docs"]+.02)
    write(OUT/"decision.json",dict(validation=means,expand=expand))
    if expand:seeds += [401,503];runseeds(seeds[-2:])
    test=load("test");domain,paths=test[2:]
    result={};raw_scores={}
    for arm in ["unmodified"]+ARMS+["intact-off"]:
        nll=[];acc=[];gates=[]
        for seed in seeds:
            basearm="host" if arm=="unmodified" else arm.split("-")[0]
            net=Surgery(basearm,seed,head.shape[1])
            if arm!="unmodified":net.load_weights(str(CACHE/f"best-{seed}-{basearm}.safetensors"))
            m,l,c,g=eval_model(net,test,off=arm=="intact-off")
            nll.append(l);acc.append(c);gates.append(g)
        l=np.stack(nll);c=np.stack(acc);g=np.stack(gates)
        raw_scores[arm]=l
        result[arm]={name:dict(nll=float(l[:,domain==k].mean()),accuracy=float(c[:,domain==k].mean()),
                          gate=float(g[:,domain==k].mean())) for k,name in enumerate(("code","docs"))}
        np.savez_compressed(OUT/(arm+"-test.npz"),nll=l,correct=c,gate=g)
    np.savez_compressed(OUT/"test-groups.npz",domain=domain,files=paths)
    contrasts={}
    for arm in ("unmodified","host","shuffled","intact-off"):
        contrasts[arm]={}
        for k,name in enumerate(("code","docs")):
            ix=domain==k;dif=raw_scores["intact"][:,ix].astype("float64")-raw_scores[arm][:,ix]
            contrasts[arm][name]=dict(mean=float(dif.mean()),interval=bootstrap_files(dif,paths[ix]),
                                      seeds=dif.mean(axis=1).tolist(),files=len(np.unique(paths[ix])))
    report=dict(results=result,intact_minus=contrasts,seeds=seeds,context=64,rank=RANK)
    write(OUT/"summary.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()
