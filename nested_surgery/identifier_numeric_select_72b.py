"""Compare a fixed donor edit at a read assignment and at a matched unused assignment."""
import argparse
from hashlib import sha256
from itertools import product
import gc
import json
import os
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, prefix
from identifier_binding import patch
from identifier_computation import options, programs
from identifier_numeric_carry_32b import verified_result
from identifier_numeric_carry_72b import ALIGNMENT_JOB, ALIGNMENT_REVISION, CUT, HOST, edits
from identifier_numeric_open import whole_vocabulary
from identifier_numeric_scale import PRIOR as MAP_JOB, archived_maps
from identifier_transfer import gather
from large_opener import MODELS, checkpoint_files, model_files

MODES=("native","selected","unused","orthogonal","whole")


def recipes():
    choices=[]
    for offset in (-3,-2,-1,1,2,3):
        for left,right in product(options(offset),repeat=2):
            original=set(left+right)
            if len(original)==4 and original.isdisjoint(v+offset for v in original):
                choices.append((offset,left,right))
    return choices


def panel(tok,count,start,*,snapshot=False):
    pool=recipes()
    rows=[]
    pair_values={name:[] for name in ("left","right")}
    for key in range(start,start+count):
        template="snapshot" if snapshot else "known-read"
        rng=random.Random(f"{template}-72b-{key}")
        offset,left,right=rng.choice(pool)
        left,right=rng.sample(left,2),rng.sample(right,2)
        for name,values in (("left",left),("right",right)):
            pair_values[name].append(values)
        x,y=f"left_{key}",f"right_{key}"
        fillers="".join(f"    temp_{i} = {rng.randrange(10,99)}\n" for i in range(rng.randrange(3,6)))
        operation=f" + {offset}" if offset>0 else f" - {-offset}"
        if snapshot:
            body=f"    {x} = {{}}\n    {y} = {x}\n{fillers}    {x} = {{}}\n"
            readers=((y,left),(x,right))
        else:
            body=f"    {x} = {{}}\n    {y} = {{}}\n{fillers}"
            readers=((x,left),(y,right))
        for chosen,(name,source) in enumerate(readers):
            prompts=[f"def fetch():\n{body.format(u,v)}    return {name}{operation}\n\nassert fetch() == "
                     for u,v in zip(left,right,strict=True)]
            answers=[str(v+offset) for v in source]
            ids=[tok.encode(p,add_special_tokens=False).ids for p in prompts]
            sites=[i for i,(a,b) in enumerate(zip(*ids,strict=True)) if a!=b]
            assert len(sites)==2 and all(len(a)==1 for a in answers)
            entire=[tok.encode(p+a,add_special_tokens=False).ids for p,a in zip(prompts,answers,strict=True)]
            assert [e[:-1] for e in entire]==ids
            for p,a in zip(prompts,answers,strict=True):
                exec(compile(p+a+"\n","<selected operand probe>","exec"),{})
            rows.append(dict(id=key,chosen=chosen,sites=sites,site=sites[chosen],prompts=prompts,ids=ids,
                             answers=answers,choices=[e[-1] for e in entire],labels=[int(a) for a in answers],
                             sources=[left,right],offset=offset))
    return rows,pair_values


def evaluate(model,rows,targets=None,geometry=None):
    records={m:[] for m in (MODES if targets is not None else ("native",))}
    doses=[]
    for i,(ids,_,ends,options,_) in enumerate(encoded_batches(rows)):
        block=rows[i*8:i*8+len(ids)//2]
        source=mx.array(np.repeat([r["sites"] for r in block],2,axis=0))
        role=mx.array(np.repeat([r["chosen"] for r in block],2,axis=0))
        index=mx.arange(len(ids))
        h=prefix(model,ids,CUT)
        records["native"].extend(whole_vocabulary(model,h,ends,CUT,options))
        if targets is not None:
            partner=index^1
            live=h[index[:,None],source].astype(mx.float32)
            target=targets[i*8:i*8+len(block)].reshape(len(ids),2,-1)[partner]
            narrow,sham=edits(live,target,*geometry)
            values=dict(selected=live[index,role]+narrow[index,role],
                        unused=live[index,1-role]+narrow[index,1-role],
                        orthogonal=live[index,role]+sham[index,role],
                        whole=live[partner,role])
            for name,replacement in values.items():
                sites=source[index,1-role] if name=="unused" else source[index,role]
                records[name].extend(whole_vocabulary(model,patch(h,sites,replacement),ends,CUT,options))
            doses.append(np.stack([np.array(mx.linalg.norm(v[index,role],axis=1)) for v in (narrow,sham)],axis=1))
            if (i+1)%8==0:
                print(f"Scored {(i+1)*8}/{len(rows)} contexts",flush=True)
    evidence={m:np.array(v) for m,v in records.items()}
    return evidence,np.concatenate(doses) if doses else None


def summarize(scores):
    native=(scores["native"][...,0]==1).reshape(-1,2,2).all(axis=(1,2))
    n=int(native.sum())
    modes={m:(a[...,0 if m=="unused" else 1]==1).reshape(-1,2,2).all(axis=(1,2))[native]
           for m,a in scores.items() if m!="native"}
    counts={m:int(b.sum()) for m,b in modes.items()}
    contrast=None
    if n:
        draws=np.random.default_rng(597620).integers(n,size=(5000,n))
        contrast=interval(modes["selected"].astype(int)-modes["orthogonal"],draws)
    passed=(n>=100 and min(counts[m] for m in ("selected","whole"))>=.8*n and counts["unused"]>=.95*n
            and contrast["mean"]>=.2 and contrast["interval"][0]>0)
    return dict(decision="selective_operand_use" if passed else "unresolved",eligible=n,counts=counts,
                selected_minus_orthogonal=contrast)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--snapshot",action="store_true")
    args=parser.parse_args()
    starts=(164000,165000,166000,167000) if args.snapshot else (135000,136000,137000,138000)
    begun=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    jobs=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    original=verified_result(jobs/ALIGNMENT_JOB,ALIGNMENT_REVISION,"cross_width_transfer")
    recipient=checkpoint_files(*HOST)
    donor_path=model_files("donor")
    tok=Tokenizer.from_file(str(recipient/"tokenizer.json"))
    donor_tok=Tokenizer.from_file(str(donor_path/"tokenizer.json"))
    screen,_=panel(tok,64,starts[0],snapshot=args.snapshot)
    rows,values=panel(tok,256,starts[1],snapshot=args.snapshot)
    model,_=load_model(recipient,strict=True)
    first,_=evaluate(model,screen)
    eligible=int((first["native"][...,0]==1).reshape(-1,2,2).all(axis=(1,2)).sum())
    np.savez_compressed(out/"screen.npz",**first)
    (out/"screen.json").write_text(json.dumps(screen,separators=(",",":"))+"\n")
    print(f"Native read-vs-unused qualification: {eligible}/64",flush=True)
    report=dict(decision="benchmark_below_floor",screen_eligible=eligible,screen_groups=64,
                models={"donor":MODELS["donor"],"recipient":HOST},cut=CUT,snapshot=args.snapshot)
    if eligible>=48:
        del model
        gc.collect()
        axes,digest=archived_maps(jobs/MAP_JOB)
        assert digest==original["map_digest"]
        donor_axis,_,_,_,affine,bias=axes
        donor,_=load_model(donor_path,strict=True)
        inputs,targets={},[]
        for name,start in zip(("left","right"),starts[2:],strict=True):
            source=programs(donor_tok,len(values[name]),start,computed=False,values_by_pair=values[name])
            inputs[name]=source
            state=mx.array(gather(donor,source,SITES["donor"]).astype(np.float32))
            coordinates=state@donor_axis@affine+bias
            targets.append(coordinates.reshape(len(source),2,-1))
        target=mx.repeat(mx.stack(targets,axis=2),2,axis=0)
        mx.eval(target)
        del donor
        gc.collect()
        stored=jobs/ALIGNMENT_JOB/"alignment.npz"
        keys=("write","read","random","center_input","center_target")
        with np.load(stored) as f:
            geometry=tuple(mx.array(f[k].astype(np.float32)) for k in keys)
        model,_=load_model(recipient,strict=True)
        records,norms=evaluate(model,rows,target,geometry)
        np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
        np.savez_compressed(out/"selection.npz",norms=norms,**records)
        (out/"programs.json").write_text(json.dumps(dict(screen=screen,host=rows,donor=inputs),separators=(",",":"))+"\n")
        report.update(groups=len(values["left"]),map_digest=digest,alignment_digest=sha256(stored.read_bytes()).hexdigest(),
                      **summarize(records))
    report.update(runtime_s=time.perf_counter()-begun,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
