"""Repeat paired identifier probes and source-token exchanges in standard checkpoints."""
import gc
import json
import os
from pathlib import Path
import random
import string

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from large_opener import MODELS, model_files

SITES = {"donor": 6, "host": 10}


def make_pairs(tokenizers, count, start, *, letter_pairs=None):
    rows = []
    for i in range(start,start+count):
        rng = random.Random(f"identifier-foundation-v1-{i}")
        stem = f"item_{100000+i}"
        letters = rng.sample(string.ascii_lowercase,2)
        if letter_pairs is not None:
            letters = letter_pairs[i-start]
        fill = "".join(f"    scratch_{k} = {rng.randrange(10,100)}\n" for k in range(rng.randint(6,14)))
        prompts = [f"def fetch():\n    {stem}{c} = 7\n{fill}    return {stem}" for c in letters]
        for p,c in zip(prompts,letters,strict=True):
            scope = {}
            exec(compile(p+c+"\n","<generated paired identifier>","exec"),scope)
            assert scope["fetch"]()==7
        encoded = []
        for tokenizer in tokenizers:
            ids = [tokenizer.encode(p,add_special_tokens=False).ids for p in prompts]
            complete = [tokenizer.encode(p+c,add_special_tokens=False).ids for p,c in zip(prompts,letters)]
            assert all(full[:-1]==prefix for full,prefix in zip(complete,ids))
            assert [tokenizer.decode(seq,skip_special_tokens=False) for seq in complete]==[p+c for p,c in zip(prompts,letters)]
            changed = [p for p,(a,b) in enumerate(zip(*ids,strict=True)) if a!=b]
            assert len(changed)==1 and complete[0][-1]!=complete[1][-1]
            encoded.append((ids,[row[-1] for row in complete],changed[0]))
        assert all(item==encoded[0] for item in encoded)
        ids,choices,site = encoded[0]
        rows.append(dict(id=i,prompts=prompts,ids=ids,choices=choices,site=site,letters=letters))
    return rows


def encoded_batches(rows):
    for i in range(0,len(rows),8):
        group = rows[i:i+8]
        seq = [s for r in group for s in r["ids"]]
        ends = np.array([len(s)-1 for s in seq])
        ids = np.zeros((len(seq),int(ends.max()+1)),np.int32)
        for j,s in enumerate(seq):
            ids[j,:len(s)] = s
        sites = np.repeat([r["site"] for r in group],2)
        candidates = np.array([c for r in group for c in r["choices"]])
        labels = np.array([ord(c)-97 for r in group for c in r["letters"]])
        yield mx.array(ids),mx.array(sites),mx.array(ends),candidates,labels


def prefix(model, ids, cut):
    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h,None)
    for layer in model.model.layers[:cut]:
        h = layer(h,mask)
    return h


def finish(model, h, ends, cut, candidates):
    mask = create_attention_mask(h,None)
    for layer in model.model.layers[cut:]:
        h = layer(h,mask)
    h = model.model.norm(h[mx.arange(len(ends)),ends])
    project = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    return odds(project(h).astype(mx.float32),candidates)


def odds(logits, candidates):
    i = mx.arange(len(candidates))
    partner = np.arange(len(candidates)) ^ 1
    return np.asarray(logits[i,mx.array(candidates)]-logits[i,mx.array(candidates[partner])]).reshape(-1,2)


def fit(x, labels):
    y = np.eye(26)[labels]
    xm,ym = x.mean(0),y.mean(0)
    c = x-xm
    gram = c.T@c
    gram += .02*np.trace(gram)/len(gram)*np.eye(len(gram))
    w = np.linalg.solve(gram,c.T@(y-ym))
    return w,ym-xm@w


def evaluate(model, calibration, test, cut):
    width = model.model.embed_tokens.weight.shape[1]
    lane = width//8
    slices = {"last":slice(width-lane,width),"adjacent":slice(width-2*lane,width-lane)}
    design, labels = [], []
    for ids,site,_,_,gold in encoded_batches(calibration):
        h = prefix(model,ids,cut)
        design.append(np.asarray(h[mx.arange(len(site)),site].astype(mx.float32)))
        labels.extend(gold)
    x = np.concatenate(design).astype(np.float64)
    maps = {name:fit(x[:,span],labels) for name,span in slices.items()}
    scores = {k:[] for k in ("native","last","adjacent","whole_site","whole_state","probe_last","probe_adjacent")}
    for i,(ids,site,end,choices,gold) in enumerate(encoded_batches(test)):
        h = prefix(model,ids,cut)
        rows = mx.arange(len(site))
        partner = mx.array(np.arange(len(site))^1)
        source = h[rows,site]
        borrowed = source[partner]
        at = (mx.arange(ids.shape[1])[None,:]==site[:,None])[...,None]
        clean = finish(model,h,end,cut,choices)
        partner_scores = finish(model,h[partner],end,cut,choices)
        np.testing.assert_array_equal(partner_scores,-clean[:,::-1])
        if i==0:
            hidden = model.model(ids)[rows,end]
            project = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
            np.testing.assert_array_equal(clean,odds(project(hidden).astype(mx.float32),choices))
        scores["native"].extend(clean)
        scores["whole_state"].extend(partner_scores)
        replace = {"whole_site":borrowed}
        for name,span in slices.items():
            keep = mx.array(np.isin(np.arange(width),np.arange(width)[span]))
            replace[name] = mx.where(keep,borrowed,source)
            w,b = maps[name]
            pred = np.asarray(source[:,span].astype(mx.float32))@w+b
            scores["probe_"+name].extend(odds(mx.array(pred.astype(np.float32)),gold))
        for name,copy in replace.items():
            scores[name].extend(finish(model,mx.where(at,copy[:,None,:],h),end,cut,choices))
    values = {k:np.asarray(v) for k,v in scores.items()}
    assert all(np.isfinite(v).all() for v in values.values())
    correct = (values["native"]>0).all(1)
    intact = int(correct.sum())
    switches = {k:int((correct&(values[k]<0).all(1)).sum()) for k in ("last","adjacent","whole_site","whole_state")}
    decoded = {k:int((values["probe_"+k]>0).all(1).sum()) for k in slices}
    reproduces = intact>=100 and min(decoded.values())>=.8*len(test) and switches["whole_site"]>=.2*intact
    reproduces = reproduces and max(switches[k] for k in slices)<.05*intact
    return values,dict(width=width,lane=lane,cut=cut,eligible=intact,switches=switches,decoded=decoded,reproduces=bool(reproduces))


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {k:model_files(k) for k in SITES}
    tokenizers = [Tokenizer.from_file(str(path/"tokenizer.json")) for path in paths.values()]
    cal,test = make_pairs(tokenizers,512,0),make_pairs(tokenizers,512,4096)
    reports = {}
    for k,cut in SITES.items():
        model,_ = load_model(paths[k],strict=True)
        values,reports[k] = evaluate(model,cal,test,cut)
        np.savez_compressed(out/(k+"-margins.npz"),**values)
        del model
        gc.collect()
    (out/"test-programs.json").write_text(json.dumps(test,indent=2)+"\n")
    (out/"result.json").write_text(json.dumps(dict(decision="complete",models=MODELS,test_pairs=len(test),calibration_pairs=len(cal),arms=reports),indent=2)+"\n")


if __name__ == "__main__":
    main()
