"""Test a projected integer edit when the two answer tokens require computation."""

import itertools
import json
import os
from pathlib import Path
import random

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES
from identifier_subspace import evaluate
from large_opener import MODELS, model_files

DIGITS = tuple(range(1,9))


def options(offset):
    return [v for v in itertools.combinations(DIGITS,2)
            if all(0<=n+offset<=9 for n in v) and set(v).isdisjoint(n+offset for n in v)]


def programs(tokenizer, count, start, *, computed, values_by_pair=None):
    output=[]
    for i in range(start,start+count):
        rng=random.Random(f"integer-computation-v1-{i}")
        offset=(1 if i%2==0 else -1)*rng.randint(1,3) if computed else 0
        pair=rng.choice(options(offset)) if computed else rng.sample(DIGITS,2)
        values=list(pair)
        rng.shuffle(values)
        if values_by_pair is not None:
            values=values_by_pair[i-start]
        rhs=f" + {offset}" if offset>0 else f" - {-offset}" if offset<0 else ""
        stem=f"amount_{900000+i}"
        scratch="".join(f"    scratch_{j} = {rng.randrange(10,99)}\n" for j in range(rng.randint(3,7)))
        prompts=[f"def fetch():\n    {stem} = {v}\n{scratch}    return {stem}{rhs}\n\nassert fetch() == " for v in values]
        answers=[str(v+offset) for v in values]
        ids=[tokenizer.encode(p,add_special_tokens=False).ids for p in prompts]
        completed=[tokenizer.encode(p+a,add_special_tokens=False).ids for p,a in zip(prompts,answers,strict=True)]
        assert [t[:-1] for t in completed]==ids
        choices=[t[-1] for t in completed]
        assert choices[0]!=choices[1]
        changed=[j for j,(a,b) in enumerate(zip(*ids,strict=True)) if a!=b]
        assert len(changed)==1
        site=changed[0]
        assert [tokenizer.decode([seq[site]]).strip() for seq in ids]==[str(v) for v in values]
        for i_answer,p in enumerate(prompts):
            for j,a in enumerate(answers):
                try:
                    exec(compile(p+a+"\n","<generated arithmetic probe>","exec"),{})
                except AssertionError:
                    assert i_answer!=j
                else:
                    assert i_answer==j
        if computed:
            assert set(answers).isdisjoint(str(v) for v in values)
        output.append(dict(id=i,ids=ids,site=site,choices=choices,labels=[v-1 for v in values],
                           prompts=prompts,source_values=values,answers=answers,offset=offset))
    return output


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    path=model_files("host")
    tok=Tokenizer.from_file(str(path/"tokenizer.json"))
    cal=programs(tok,512,0,computed=False)
    test=programs(tok,512,60000,computed=True)
    model,_=load_model(path,strict=True)
    scores,norms,arm=evaluate(model,cal,test,SITES["host"],classes=len(DIGITS))
    n=arm["eligible"]
    contrast=arm["learned_minus_matched"]
    passed=(n>=100 and arm["switches"]["whole_site"]>=.6*n and arm["switches"]["learned"]>=.6*n
            and contrast["mean"]>=.2 and contrast["interval"][0]>0)
    np.savez_compressed(out/"computed.npz",norms=norms,**scores)
    (out/"programs.json").write_text(json.dumps(dict(calibration=cal,test=test),separators=(",",":"))+"\n")
    record=dict(decision="computed_transfer" if passed else "unresolved",model=MODELS["host"],groups=len(test),arm=arm)
    (out/"result.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
