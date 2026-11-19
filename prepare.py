"""Prepare pinned public models, file-disjoint samples, and frozen branch features."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from mlx_lm.models.qwen2 import Model, ModelArgs
from mlx_lm.models.base import create_attention_mask

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/"cache"
OUT=ROOT/"results"
MODELS={"host":("Qwen/Qwen2.5-1.5B","8faed761d45a263340a0528343f099c05c9a4323"),
        "donor":("Qwen/Qwen2.5-Coder-0.5B","8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301")}
CPYTHON="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
POS=np.array([31,39,47,55],dtype=np.int32)


def write(path,obj):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2)+"\n");tmp.replace(path)


def fetch():
    CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    for dest,(model,rev) in MODELS.items():
        snapshot_download(model,revision=rev,local_dir=CACHE/dest,
                          allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json",
                                          "generation_config.json","merges.txt","vocab.json"])
    tar=CACHE/"cpython.tar.gz"
    if not tar.exists():
        urllib.request.urlretrieve(f"https://codeload.github.com/python/cpython/tar.gz/{CPYTHON}",tar)


def samples():
    tok=Tokenizer.from_file(str(CACHE/"host/tokenizer.json"))
    other=Tokenizer.from_file(str(CACHE/"donor/tokenizer.json"))
    pools={};rng=np.random.default_rng(54329)
    with tarfile.open(CACHE/"cpython.tar.gz") as archive:
        for entry in sorted(archive.getmembers(),key=lambda x:x.name):
            path=entry.name.partition("/")[2]
            if not entry.isfile() or entry.size>1_500_000:continue
            if path.startswith("Lib/") and path.endswith(".py") and not set(path.split("/"))&{"test","tests","ensurepip"}:domain=0
            elif path.startswith("Doc/") and path.endswith(".rst"):domain=1
            else:continue
            bucket=int.from_bytes(hashlib.sha256(path.encode()).digest()[:4],"little")%100
            split="train" if bucket<65 else ("val" if bucket<80 else "test")
            text=archive.extractfile(entry).read().decode("utf-8",errors="replace")
            ids=tok.encode(text,add_special_tokens=False).ids
            if len(ids)<65:continue
            if ids!=other.encode(text,add_special_tokens=False).ids:continue
            offsets=np.arange(0,len(ids)-64,65)
            rng.shuffle(offsets)
            pools.setdefault((domain,split),[]).append((path,ids,offsets[:24]))
    seen=set();manifest=[]
    for split,n in [("train",768),("val",128),("test",512)]:
        windows=[];domains=[];paths=[]
        for domain in [0,1]:
            pool=pools[domain,split];order=rng.permutation(len(pool))
            count=0
            for round_ in range(24):
                for i in order:
                    path,ids,possible=pool[i]
                    if round_>=len(possible):continue
                    off=int(possible[round_])
                    seq=np.array(ids[off:off+65],dtype="int32")
                    digest=hashlib.sha256(seq.tobytes()).hexdigest()
                    if digest in seen:continue
                    seen.add(digest);windows.append(seq);domains.append(domain);paths.append(path)
                    manifest.append([split,domain,path,off,digest]);count+=1
                    if count==n:break
                if count==n:break
            assert count==n,(domain,split,count)
        np.savez(CACHE/(split+"-tokens.npz"),tokens=np.stack(windows),domain=domains,path=paths)
    sets=[{row[2] for row in manifest if row[0]==s} for s in ("train","val","test")]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    write(OUT/"windows.json",manifest)
    print("file-disjoint corpus",[(s,len(t)) for s,t in zip(("train","val","test"),sets)],flush=True)


def load_model(name):
    path=CACHE/name
    args=ModelArgs.from_dict(json.loads((path/"config.json").read_text()))
    model=Model(args)
    weights=model.sanitize(mx.load(str(path/"model.safetensors")))
    model.load_weights(list(weights.items()),strict=True)
    model.eval();mx.eval(model.parameters())
    return model


def features(model,ids):
    core=model.model;x=core.embed_tokens(ids)
    mask=create_attention_mask(x,None)
    for block in core.layers[:-1]:x=block(x,mask)
    last=core.layers[-1]
    h=x+last.self_attn(last.input_layernorm(x),mask)
    h=h[:,mx.array(POS),:]
    u=last.post_attention_layernorm(h);f=last.mlp(u)
    return u,f,h+f


def cache_states(name):
    model=load_model(name)
    for split in (("train",) if name=="donor" else ("train","val","test")):
        with np.load(CACHE/(split+"-tokens.npz")) as d: raw=d["tokens"]
        groups=[[],[],[]]
        for off in range(0,len(raw),8):
            x=mx.array(raw[off:off+8,:64])
            states=features(model,x)
            if name=="host" and off==0 and split=="train":
                logits=model(x[:1])[:,mx.array(POS)]
                cached=model.model.embed_tokens.as_linear(model.model.norm(states[2][:1]))
                labels=mx.array(raw[:1,POS+1])
                err=float(mx.max(mx.abs(nn.losses.cross_entropy(logits.astype(mx.float32),labels)-
                                        nn.losses.cross_entropy(cached.astype(mx.float32),labels))).item())
                assert err<.05,err
                write(OUT/"host-check.json",{"max_loss_error":err,"top1_equal":bool(mx.all(logits.argmax(-1)==cached.argmax(-1)).item())})
            for j,arr in enumerate(states[:1] if name=="donor" else states):
                groups[j].append(np.array(arr.astype(mx.float32)).reshape(-1,arr.shape[-1]).astype("float16"))
        np.savez(CACHE/(name+"-"+split+".npz"),**{k:np.concatenate(z) for k,z in zip(("u","native","base"),groups) if z})
        print("cached",name,split,len(raw),flush=True)
    return model


def ridge(x,y):
    x=x.astype("float32");y=y.astype("float32")
    a=x.mean(axis=0);b=y.mean(axis=0);x-=a;y-=b
    gram=x.T@x
    lam=.02*np.trace(gram)/len(gram)
    gram.flat[::len(gram)+1]+=lam
    w=np.linalg.solve(gram,x.T@y)
    return w,b-a@w


def branches(donor):
    block=donor.model.layers[-1].mlp
    paired=np.random.default_rng(9917).permutation(block.up_proj.weight.shape[0])
    shuffled=block.up_proj.weight[mx.array(paired)]
    with np.load(CACHE/"host-train.npz") as h,np.load(CACHE/"donor-train.npz") as d:
        win,bin_=ridge(h["u"],d["u"])
    fits={}
    for split in ("train","val","test"):
        with np.load(CACHE/("host-"+split+".npz")) as z:
            host_u=z["u"];native=z["native"].astype("float32")
        projected=(host_u.astype("float32")@win+bin_).astype("float16")
        arrays={}
        for arm in ("intact","shuffled"):
            parts=[]
            for off in range(0,len(projected),128):
                x=mx.array(projected[off:off+128]).astype(mx.bfloat16)
                y=block(x) if arm=="intact" else block.down_proj(nn.silu(block.gate_proj(x))*(x@shuffled.T))
                parts.append(np.array(y.astype(mx.float32)))
            raw=np.concatenate(parts)
            if split=="train":fits[arm]=ridge(raw,native)
            w,bias=fits[arm]
            arrays[arm]=(raw@w+bias-native).astype("float16")
        np.savez(CACHE/("branches-"+split+".npz"),**arrays)
    print("training-only bridges fitted for both donors",flush=True)


if __name__=="__main__":
    mx.set_default_device(mx.gpu);fetch();samples()
    host=cache_states("host");del host;mx.clear_cache()
    donor=cache_states("donor");branches(donor)
    write(OUT/"provenance.json",{"models":MODELS,"cpython":CPYTHON,"device":mx.device_info()["device_name"],
         "positions":POS.tolist(),"donor_layer":23,"host_layer":27,"shuffle_seed":9917,"ridge_trace_factor":.02})
