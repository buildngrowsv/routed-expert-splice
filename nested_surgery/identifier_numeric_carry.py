"""Verify greedy addition completions whose length changes under operand interventions."""
import gc
from itertools import combinations, product
import json
import os
from pathlib import Path
import random

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, prefix
from identifier_computation import DIGITS, programs
from identifier_numeric_composition import MAP_JOB, load_prior, replace
from identifier_transfer import gather, transform
from large_opener import MODELS, model_files

MODES=("native","both","random","whole")


def carry_recipes():
    pairs=list(combinations(DIGITS,2))
    recipes=[]
    for a,b in product(pairs,repeat=2):
        outputs={x+y for x,y in product(a,b)}
        if len(outputs)==4 and min(outputs)<10<=max(outputs) and outputs.isdisjoint(a+b):
            recipes.append((a,b))
    return recipes


def host_panel(tokenizer,count,start):
    recipes=carry_recipes()
    rows=[]
    for index in range(start,start+count):
        rng=random.Random(f"numeric-carry-v1-{index}")
        a,b=rng.choice(recipes)
        left,right=rng.sample(a,2),rng.sample(b,2)
        values=list(product(left,right))
        x,y=f"left_{800000+index}",f"right_{800000+index}"
        filler="".join(f"    scratch_{j} = {rng.randrange(10,99)}\n" for j in range(rng.randint(3,7)))
        tail=f"{filler}    return {x} + {y}\n\nassert fetch() == "
        prompts=[f"def fetch():\n    {x} = {u}\n    {y} = {v}\n"+tail for u,v in values]
        answers=[str(u+v) for u,v in values]
        ids=[tokenizer.encode(p,add_special_tokens=False).ids for p in prompts]
        sites=[pos for pos,z in enumerate(zip(*ids,strict=True)) if len(set(z))>1]
        assert len(sites)==2
        digits=None
        for variant,(text,start_ids) in enumerate(zip(prompts,ids,strict=True)):
            encoded=[]
            for inspected,value in enumerate(answers):
                all_ids=tokenizer.encode(text+value,add_special_tokens=False).ids
                np.testing.assert_array_equal(all_ids[:len(start_ids)],start_ids)
                suffix=all_ids[len(start_ids):]
                assert [tokenizer.decode([i],skip_special_tokens=False) for i in suffix]==list(value)
                encoded.append(suffix)
                try:
                    exec(compile(text+value+"\n","<generated carry assertion>","exec"),{})
                except AssertionError:
                    assert variant!=inspected
                else:
                    assert variant==inspected
            if digits is None:
                digits=encoded
            assert digits==encoded
        rows.append(dict(id=index,left=left,right=right,ids=ids,sites=sites,
                         answers=answers,answer_tokens=digits,prompts=prompts))
    return rows


def decoded_steps(model,h,positions,cut):
    mask=create_attention_mask(h,None)
    for layer in model.model.layers[cut:]:
        h=layer(h,mask)
    state=model.model.norm(h[mx.arange(h.shape[0])[:,None],positions])
    head=model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    scores=head(state).astype(mx.float32)
    top=mx.argmax(scores,axis=-1)
    unique=mx.sum(scores==scores.max(-1,keepdims=True),axis=-1)==1
    return np.array(mx.stack((top,unique.astype(mx.int32)),axis=-1)).reshape(-1,4,3,2)


def predict(model,rows,changes=None):
    cut=SITES["host"]
    outputs={}
    for start in range(0,len(rows),4):
        block=rows[start:start+4]
        sites=mx.array(np.repeat([r["sites"] for r in block],4,axis=0))
        for mode in MODES if changes is not None else ("native",):
            target_index=np.arange(4) if mode=="native" else np.arange(4)[::-1]
            original=[ids for row in block for ids in row["ids"]]
            target=[row["answer_tokens"][v] for row in block for v in target_index]
            seq=[ids+suffix for ids,suffix in zip(original,target,strict=True)]
            max_len=max(map(len,seq))
            positions=[list(range(len(a)-1,len(a)+len(b)))+[len(a)+len(b)-1]*(2-len(b))
                       for a,b in zip(original,target,strict=True)]
            h=prefix(model,mx.array([a+[0]*(max_len-len(a)) for a in seq]),cut)
            if mode!="native":
                index=mx.arange(len(seq))
                state=h[index[:,None],sites].astype(mx.float32)
                if mode=="whole":
                    values=state[index^3].transpose(1,0,2)
                else:
                    which=1 if mode=="random" else 0
                    rows_slice=slice(start,start+len(block))
                    values=[]
                    for i,(name,pattern) in enumerate((("left",(0,0,1,1)),("right",(0,1,0,1)))):
                        edit=changes[name][which][rows_slice][:,pattern].reshape(len(seq),-1)
                        values.append(state[:,i]+mx.array(edit))
                h=replace(h,sites,values)
            outputs.setdefault(mode,[]).extend(decoded_steps(model,h,mx.array(positions),cut))
        if changes is not None and (start+len(block))%64==0:
            print(f"Scored {start+len(block)}/{len(rows)} groups",flush=True)
    return {m:np.array(v) for m,v in outputs.items()}


def finish_tokens(tokenizer,values):
    ends={tokenizer.token_to_id(s) for s in ("<|endoftext|>","<|im_end|>")}
    return {int(i) for i in values if int(i) in ends or
            tokenizer.decode([int(i)],skip_special_tokens=False).lstrip(" \t").startswith(("\n","\r","#"))}


def outcomes(rows,outputs,terminators):
    result={}
    for mode,records in outputs.items():
        order=range(4) if mode=="native" else reversed(range(4))
        indices=tuple(order)
        correct=np.zeros((len(rows),4),dtype=bool)
        for i,row in enumerate(rows):
            for j,target in enumerate(indices):
                digits=row["answer_tokens"][target]
                k=len(digits)
                z=records[i,j]
                correct[i,j]=(z[:k,0].tolist()==digits and z[:k,1].all() and int(z[k,0]) in terminators)
        result[mode]=correct
    return result


def summarize(rows,hits):
    eligible=hits["native"].all(1)
    n=int(eligible.sum())
    won={m:a[eligible].all(1) for m,a in hits.items()}
    counts={m:int(a.sum()) for m,a in won.items()}
    lengths=np.array([[len(a) for a in row["answers"]] for row in rows])
    changes={"one_to_two":(lengths==1)&(lengths[:,::-1]==2),
             "two_to_one":(lengths==2)&(lengths[:,::-1]==1)}
    crossed={}
    for name,mask in changes.items():
        good=mask&eligible[:,None]
        crossed[name]=dict(prompts=int(good.sum()),correct={m:int(a[good].sum()) for m,a in hits.items() if m!="native"})
    effect=None
    if n:
        d=won["both"].astype(int)-won["random"]
        draw=np.random.default_rng(814225).integers(n,size=(5000,n))
        effect=interval(d,draw)
    passed=(n>=100 and min(counts["both"],counts["whole"])>=.8*n
            and effect["mean"]>=.2 and effect["interval"][0]>0)
    return dict(decision="carry_composes" if passed else "unresolved",eligible=n,correct=counts,
                direction_counts=crossed,both_minus_random=effect)


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    tok=Tokenizer.from_file(str(model_files("host")/"tokenizer.json"))
    screening=host_panel(tok,128,110000)
    host,_=load_model(model_files("host"),strict=True)
    first=predict(host,screening)
    delim=finish_tokens(tok,np.unique(first["native"][...,0]))
    eligible=int(outcomes(screening,first,delim)["native"].all(1).sum())
    np.savez_compressed(out/"screen.npz",terminators=np.array(sorted(delim)),**first)
    (out/"screen.json").write_text(json.dumps(screening,separators=(",",":"))+"\n")
    report=dict(decision="benchmark_below_floor",screen_eligible=eligible,screen_groups=128,models=MODELS)
    print(f"Native carry qualification {eligible}/128",flush=True)
    if eligible>=64:
        del host
        gc.collect()
        panel=host_panel(tok,512,111000)
        prior=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/MAP_JOB
        axes,digest=load_prior(prior)
        donor_tok=Tokenizer.from_file(str(model_files("donor")/"tokenizer.json"))
        donor,_=load_model(model_files("donor"),strict=True)
        sources,changes,norms={},{},{}
        for side,start in (("left",112000),("right",113000)):
            src=programs(donor_tok,len(panel),start,computed=False,values_by_pair=[r[side] for r in panel])
            sources[side]=src
            state=gather(donor,src,SITES["donor"]).astype(np.float32)
            changes[side]=[np.array(a).reshape(len(panel),2,-1) for a in transform(mx.array(state),*axes)]
            norms[side]=np.stack([np.linalg.norm(a,axis=2) for a in changes[side]],axis=-1)
        del donor
        gc.collect()
        host,_=load_model(model_files("host"),strict=True)
        scores=predict(host,panel,changes)
        assert all(a.shape==(512,4,3,2) for a in scores.values())
        endings=finish_tokens(tok,np.unique(np.concatenate([v[...,0].ravel() for v in scores.values()])))
        report.update(groups=len(panel),map_digest=digest,**summarize(panel,outcomes(panel,scores,endings)))
        np.savez_compressed(out/"carry.npz",left_norms=norms["left"],right_norms=norms["right"],
                            terminators=np.array(sorted(endings)),**scores)
        (out/"programs.json").write_text(json.dumps(dict(host=panel,donor=sources),separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
