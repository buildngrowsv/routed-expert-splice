"""Forced last-MLP replacement. No host MLP output is an input to the graft."""
import hashlib
import json
import tarfile
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tokenizers import Tokenizer
from prepare import CACHE, OUT, POS, load_model, create_attention_mask, ridge, write
from run import bootstrap_files

RES=OUT/"forced"
SEEDS=[101,211,307]
ARMS=["intact","shuffled"]


def new_test(previous=None, seed=91003, name="fresh", output=None):
    manifest=json.loads((OUT/"windows.json").read_text())
    used={r[4] for r in manifest};offsets={};allowed={}
    for split,domain,path,off,_ in manifest:
        if split=="test":
            offsets.setdefault(path,set()).add(off);allowed[path]=domain
    if previous is not None:
        for domain,path,off,digest in json.loads(previous.read_text()):
            used.add(digest)
            offsets.setdefault(path,set()).add(off)
    tok=Tokenizer.from_file(str(CACHE/"host/tokenizer.json"))
    other=Tokenizer.from_file(str(CACHE/"donor/tokenizer.json"))
    groups=[[],[]];rng=np.random.default_rng(seed)
    with tarfile.open(CACHE/"cpython.tar.gz") as tar:
        for entry in sorted(tar.getmembers(),key=lambda e:e.name):
            path=entry.name.partition("/")[2]
            if path not in allowed:continue
            source=tar.extractfile(entry).read().decode("utf8",errors="replace")
            ids=tok.encode(source,add_special_tokens=False).ids
            assert ids==other.encode(source,add_special_tokens=False).ids
            starts=[v for v in range(0,len(ids)-64,65) if v not in offsets[path]]
            rng.shuffle(starts)
            if starts:groups[allowed[path]].append((path,ids,starts))
    seq=[];records=[]
    for domain,pool in enumerate(groups):
        count=0;order=rng.permutation(len(pool))
        for turn in range(max(len(p[2]) for p in pool)):
            for i in order:
                path,ids,options=pool[i]
                if turn>=len(options):continue
                off=options[turn];win=np.array(ids[off:off+65],dtype="int32")
                digest=hashlib.sha256(win.tobytes()).hexdigest()
                if digest in used:continue
                used.add(digest);seq.append(win);records.append([domain,path,off,digest]);count+=1
                if count==512:break
            if count==512:break
        assert count==512
    np.savez(CACHE/(name+"-tokens.npz"),tokens=np.stack(seq),domain=[r[0] for r in records],path=[r[1] for r in records])
    write(output if output is not None else RES/"windows.json",records)


def states(model,ids):
    core=model.model;x=core.embed_tokens(ids);mask=create_attention_mask(x,None)
    for block in core.layers[:-1]:x=block(x,mask)
    last=core.layers[-1];h=x+last.self_attn(last.input_layernorm(x),mask)
    u=last.post_attention_layernorm(h);native=last.mlp(u)
    return h,u,native,h+native


class Connector(nn.Module):
    def __init__(self,seed,d):
        super().__init__()
        rng=np.random.default_rng(seed)
        self.left=mx.array(rng.normal(0,d**-.5,(d,32)).astype("float32"))
        self.right=mx.zeros((32,d))
    def __call__(self,u):
        return nn.silu(mx.fast.rms_norm(u.astype(mx.float32),None,1e-5)@self.left)@self.right


class Replacement(nn.Module):
    def __init__(self,donor_mlp,arm,win,bin_,wout,bout,connector):
        super().__init__()
        self.expert=donor_mlp;self.arm=arm;self.win=win;self.bin=bin_
        self.wout=wout;self.bout=bout;self.connector=connector
        ix=np.random.default_rng(9917).permutation(donor_mlp.up_proj.weight.shape[0])
        self.shuffled_up=donor_mlp.up_proj.weight[mx.array(ix)]
    def expert_output(self,u):
        z=(u.astype(mx.float32)@self.win+self.bin).astype(mx.bfloat16)
        if self.arm=="intact":return self.expert(z)
        return self.expert.down_proj(nn.silu(self.expert.gate_proj(z))*(z@self.shuffled_up.T))
    def __call__(self,u):
        frozen=self.expert_output(u).astype(mx.float32)@self.wout+self.bout
        return (frozen+self.connector(u)).astype(mx.bfloat16)


def main():
    mx.set_default_device(mx.gpu);RES.mkdir(parents=True,exist_ok=True)
    new_test();host=load_model("host");donor=load_model("donor");core=host.model
    def logits(h):return core.embed_tokens.as_linear(core.norm(h.astype(mx.bfloat16))).astype(mx.float32)
    tables={}
    for split in ("train","val","fresh"):
        with np.load(CACHE/(split+"-tokens.npz")) as z:
            ids=z["tokens"];dom=np.repeat(z["domain"],len(POS));files=np.repeat(z["path"],len(POS))
        chunks=[[] for _ in range(4)];du=[]
        for off in range(0,len(ids),8):
            x=mx.array(ids[off:off+8,:64])
            st=states(host,x)
            if split=="train":du.append(np.array(states(donor,x)[1][:,mx.array(POS)].astype(mx.float32)).reshape(-1,896))
            for c,t in zip(chunks,st):
                c.append(np.array(t[:,mx.array(POS)].astype(mx.float32)).reshape(-1,t.shape[-1]))
        tables[split]={k:mx.array(np.concatenate(z)) for k,z in zip(("h","u","ff","base"),chunks)}
        tables[split].update(y=mx.array(ids[:,POS+1].reshape(-1)),domain=dom,files=files)
        if du:donor_train=np.concatenate(du)
    tr=tables["train"];win,bi=ridge(np.array(tr["u"]),donor_train)
    win,bi=mx.array(win),mx.array(bi)
    bridges={};fits={};raw={}
    for arm in ARMS:
        rep=Replacement(donor.model.layers[-1].mlp,arm,win,bi,None,None,Connector(0,1536))
        for split,t in tables.items():
            v=mx.concatenate([rep.expert_output(t["u"][i:i+128]).astype(mx.float32) for i in range(0,len(t["y"]),128)])
            if split=="train":
                wo,bo=ridge(np.array(v),np.array(t["ff"]));bridges[arm]=mx.array(wo),mx.array(bo)
            t[arm]=v@bridges[arm][0]+bridges[arm][1];mx.eval(t[arm])
    def eval_(t,mode,net=None,no_expert=False):
        losses=[];hits=[]
        for i in range(0,len(t["y"]),96):
            sl=slice(i,i+96)
            if mode=="native":h=t["base"][sl]
            elif mode=="knockout":h=t["h"][sl]
            else:
                y=net(t["u"][sl]) if net else 0.
                if not no_expert:y=y+t[mode][sl]
                h=t["h"][sl].astype(mx.bfloat16)+y.astype(mx.bfloat16)
            l=logits(h);target=t["y"][sl]
            losses.append(np.array(nn.losses.cross_entropy(l,target,reduction="none")))
            hits.append(np.array(l.argmax(1)==target))
        return np.concatenate(losses),np.concatenate(hits)
    def pack(l,h,t):
        return {name:dict(nll=float(l[...,t["domain"]==k].mean()),accuracy=float(h[...,t["domain"]==k].mean()))
                for k,name in enumerate(("code","docs"))}
    for arm in ARMS:
        for seed in SEEDS:
            net=Connector(seed,1536);opt=optim.Adam(.002)
            def obj(m,u,expert,teacher):
                residual=expert+m(u)-teacher
                return ((residual*residual).mean(-1)/((teacher*teacher).mean(-1)+.01)).mean()
            vg=nn.value_and_grad(net,obj);st=[net.state,opt.state]
            @partial(mx.compile,inputs=st,outputs=st)
            def step(u,expert,teacher):
                loss,g=vg(net,u,expert,teacher);opt.update(net,g);return loss
            best=float("inf");history=[];rng=np.random.default_rng(12000+seed)
            ck=CACHE/f"forced-{arm}-{seed}.safetensors"
            for ep in range(9):
                if ep:
                    order=mx.array(rng.permutation(len(tr["y"])).astype("int32"))
                    for i in range(0,len(order),96):
                        ix=order[i:i+96];loss=step(tr["u"][ix],tr[arm][ix],tr["ff"][ix]);mx.eval(loss,st)
                if ep in (0,1,2,4,8):
                    l,h=eval_(tables["val"],arm,net)
                    m=pack(l,h,tables["val"]);history.append([ep,m]);score=l.mean()
                    if score<best:
                        best=score;chosen=ep;net.save_weights(str(ck))
            fits[f"{seed}-{arm}"]={"selected":chosen,"validation":history}
            print("forced",arm,seed,"epoch",chosen,flush=True)
    write(RES/"selection.json",fits)
    t=tables["fresh"];report={};native_losses={}
    for name in ("native","knockout","intact-fit","shuffled-fit","intact","shuffled","intact-off","shuffled-off"):
        larr=[];harr=[]
        for seed in SEEDS:
            mode=name.split("-")[0];net=None
            if mode in ARMS and not name.endswith("-fit"):
                net=Connector(seed,1536);net.load_weights(str(CACHE/f"forced-{mode}-{seed}.safetensors"))
            l,h=eval_(t,mode,net,name.endswith("-off"));larr.append(l);harr.append(h)
        ls=np.stack(larr);hs=np.stack(harr);native_losses[name]=ls
        report[name]=pack(ls,hs,t);np.savez_compressed(RES/(name+".npz"),nll=ls,correct=hs)
    # Actually replace the module and call the stock model's full forward path.
    net=Connector(SEEDS[0],1536);net.load_weights(str(CACHE/"forced-intact-101.safetensors"))
    graft=Replacement(donor.model.layers[-1].mlp,"intact",win,bi,*bridges["intact"],net)
    with np.load(CACHE/"fresh-tokens.npz") as z:ids=mx.array(z["tokens"][:8,:64])
    last=core.layers[-1];original=last.mlp
    try:
        last.mlp=graft
        live=host(ids)[:,mx.array(POS)].reshape(32,-1).astype(mx.float32)
        loss=np.array(nn.losses.cross_entropy(live,t["y"][:32],reduction="none"))
        actual=np.array(live.argmax(-1))
    finally:last.mlp=original
    proxy=logits(t["h"][:32].astype(mx.bfloat16)+(t["intact"][:32]+net(t["u"][:32])).astype(mx.bfloat16))
    ref=np.array(nn.losses.cross_entropy(proxy,t["y"][:32],reduction="none"))
    check=dict(tokens=32,top1_agreement=float((actual==np.array(proxy.argmax(-1))).mean()),
               mean_abs_nll=float(np.abs(loss-ref).mean()),max_abs_nll=float(np.abs(loss-ref).max()))
    assert check["top1_agreement"]>=.95 and check["mean_abs_nll"]<.03,check
    np.savez_compressed(RES/"test-groups.npz",domain=t["domain"],files=t["files"])
    contrasts={}
    for other in ("native","knockout","shuffled","intact-fit","intact-off"):
        contrasts[other]={}
        for k,n in enumerate(("code","docs")):
            ix=t["domain"]==k
            dif=native_losses["intact"][:,ix].astype("float64")-native_losses[other][:,ix]
            contrasts[other][n]=dict(mean=float(dif.mean()),interval=bootstrap_files(dif,t["files"][ix]))
    recovery={k:(report["knockout"][k]["nll"]-report["intact"][k]["nll"])/
                 (report["knockout"][k]["nll"]-report["native"][k]["nll"]) for k in ("code","docs")}
    write(RES/"summary.json",dict(results=report,intact_minus=contrasts,recovery=recovery,
                                  live_replacement_check=check,seeds=SEEDS))
    print("forced results",report,"recovery",recovery,"parity",check,flush=True)


if __name__=="__main__":main()
