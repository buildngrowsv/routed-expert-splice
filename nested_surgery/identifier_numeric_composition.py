"""Compose two independently translated operands in four-choice Python arithmetic."""

import gc
from hashlib import sha256
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
from identifier_transfer import gather, transform
from large_opener import MODELS, model_files

ARMS = {"native": 0, "left": 2, "right": 1, "both": 3, "random": 3, "whole": 3}
MAP_JOB = "qwen-numeric-operand-cross-model"
MAP_REV = "3eacf5424a46a4e2efb328de5fb0ad77f391bc15"


def recipes(sign):
    result = []
    pairs = list(combinations(DIGITS, 2))
    for left, right in product(pairs, repeat=2):
        initial = [a + sign * 2 * b for a, b in product(left, right)]
        if len(set(initial)) != 4:
            continue
        for shift in range(-min(initial), 10 - max(initial)):
            answers = {n + shift for n in initial}
            if answers.isdisjoint((*left, *right)):
                result.append((left, right, shift))
    return result


def host_panel(tokenizer, count=1024, start=80000):
    source = {sign: recipes(sign) for sign in (-1, 1)}
    groups = []
    for index in range(start, start + count):
        rng = random.Random(f"numeric-composition-v1-{index}")
        sign = 1 if index % 2 == 0 else -1
        a, b, shift = rng.choice(source[sign])
        left, right = rng.sample(a, 2), rng.sample(b, 2)
        pairs = list(product(left, right))
        x, y = f"left_{900000+index}", f"right_{900000+index}"
        scratch = "".join(f"    scratch_{k} = {rng.randrange(10,99)}\n" for k in range(rng.randint(3,7)))
        expression = f"{x} {'+' if sign>0 else '-'} 2 * {y} {'+' if shift>=0 else '-'} {abs(shift)}"
        prompts = [f"def fetch():\n    {x} = {u}\n    {y} = {v}\n{scratch}    return {expression}\n\nassert fetch() == "
                   for u, v in pairs]
        answers = [str(u + sign * 2 * v + shift) for u, v in pairs]
        ids = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
        full = [tokenizer.encode(p+c, add_special_tokens=False).ids for p,c in zip(prompts,answers,strict=True)]
        assert [s[:-1] for s in full] == ids
        choices = [s[-1] for s in full]
        assert len(set(choices)) == 4 and set(map(int,answers)).isdisjoint((*left,*right))
        sites = [j for j, values in enumerate(zip(*ids, strict=True)) if len(set(values)) > 1]
        assert len(sites) == 2
        for state,ids_in_prompt in enumerate(ids):
            assert [tokenizer.decode([ids_in_prompt[j]]).strip() for j in sites] == list(map(str,pairs[state]))
        for state,prompt in enumerate(prompts):
            for choice,answer in enumerate(answers):
                try:
                    exec(compile(prompt+answer+"\n","<generated two-operand probe>","exec"),{})
                except AssertionError:
                    assert choice != state
                else:
                    assert choice == state
        groups.append(dict(id=index, ids=ids, sites=sites, choices=choices, prompts=prompts,
                           answers=answers, left=left, right=right, sign=sign, shift=shift))
    return groups


def readout(model, h, ends, cut, choices):
    mask = create_attention_mask(h,None)
    for block in model.model.layers[cut:]:
        h = block(h,mask)
    index = mx.arange(h.shape[0])
    out = model.model.norm(h[index,ends])
    project = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    return np.asarray(project(out)[index[:,None],choices].astype(mx.float32)).reshape(-1,4,4)


def replace(h, locations, vectors):
    idx = mx.arange(h.shape[1])[None,:]
    for i, values in enumerate(vectors):
        at = (idx==locations[:,i,None])[...,None]
        h = mx.where(at,values[:,None,:].astype(h.dtype),h)
    return h


def evaluate(model, panel, changes):
    cut = SITES["host"]
    raw = {k:[] for k in ARMS}
    for start in range(0,len(panel),4):
        block = panel[start:start+4]
        seqs = [seq for row in block for seq in row["ids"]]
        ends = mx.array([len(s)-1 for s in seqs])
        width = max(map(len,seqs))
        tokens = mx.array([s+[0]*(width-len(s)) for s in seqs])
        choices = mx.array(np.repeat([r["choices"] for r in block],4,axis=0))
        sites = mx.array(np.repeat([r["sites"] for r in block],4,axis=0))
        index = mx.arange(len(seqs))
        h = prefix(model,tokens,cut)
        state = h[index[:,None],sites].astype(mx.float32)
        state_left,state_right = state[:,0],state[:,1]
        deltas = []
        for side, pattern in (("left",(0,0,1,1)),("right",(0,1,0,1))):
            d,r = changes[side]
            subset = slice(start,start+len(block))
            deltas.append([mx.array(v[subset][:,pattern].reshape(len(seqs),-1)) for v in (d,r)])
        (left,left_random),(right,right_random) = deltas
        altered = {
            "native": h,
            "left": replace(h,sites,(state_left+left,state_right)),
            "right": replace(h,sites,(state_left,state_right+right)),
            "both": replace(h,sites,(state_left+left,state_right+right)),
            "random": replace(h,sites,(state_left+left_random,state_right+right_random)),
            "whole": replace(h,sites,state[index ^ 3].transpose(1,0,2)),
        }
        for name, alt in altered.items():
            raw[name].extend(readout(model,alt,ends,cut,choices))
        if (start+len(block))%64==0:
            print(f"Scored {start+len(block)}/{len(panel)} groups",flush=True)
    return {k:np.asarray(v) for k,v in raw.items()}


def summarize(scores, panel):
    states = np.arange(4)
    outcomes = {}
    for name, arm in scores.items():
        targets = (states ^ ARMS[name])[None,:,None]
        target = np.take_along_axis(arm,targets,axis=2)[...,0]
        other = np.max(np.where(states[None,None,:]==targets,-np.inf,arm),axis=2)
        outcomes[name] = (target>other).all(axis=1)
    eligible = outcomes["native"]
    n = int(eligible.sum())
    counts = {k:int(v[eligible].sum()) for k,v in outcomes.items()}
    by_operator = {}
    sign = np.asarray([row["sign"] for row in panel])
    for op, condition in (("addition",sign>0),("subtraction",sign<0)):
        subset = eligible & condition
        by_operator[op] = dict(eligible=int(subset.sum()),cells={k:int(v[subset].sum()) for k,v in outcomes.items()})
    contrast = None
    if n:
        idx = np.random.default_rng(853143).integers(n,size=(5000,n))
        contrast = interval(outcomes["both"][eligible].astype(int)-outcomes["random"][eligible],idx)
    passed = (n>=100 and min(counts[k] for k in ("whole","left","right","both"))>=.6*n
              and contrast["interval"][0]>.2)
    return dict(decision="composes" if passed else "unresolved", eligible=n, cells=counts,
                both_minus_random=contrast, by_operator=by_operator)


def load_prior(run):
    receipt = json.loads((run/"receipt.json").read_text())
    assert receipt["revision"] == MAP_REV and receipt["status"] == "done"
    assert json.loads((run/"result.json").read_text())["decision"] == "transfer"
    archive = run/"map.npz"
    with np.load(archive) as z:
        raw = [z[k] for k in ("donor_axis","host_axis","random_host_axis","mapping")]
    assert raw[-1].shape == (7,7)
    return [mx.array(x) for x in raw],sha256(archive.read_bytes()).hexdigest()


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    paths = {name:model_files(name) for name in SITES}
    tok = {k:Tokenizer.from_file(str(v/"tokenizer.json")) for k,v in paths.items()}
    axes,digest = load_prior(Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/MAP_JOB)
    panel = host_panel(tok["host"])
    donor,_ = load_model(paths["donor"],strict=True)
    sources,changes,norms = {},{},{}
    for side,start in (("left",84000),("right",86000)):
        source = programs(tok["donor"],len(panel),start,computed=False,values_by_pair=[r[side] for r in panel])
        sources[side] = source
        donor_states = gather(donor,source,SITES["donor"])
        delta,random_edit = transform(mx.array(donor_states.astype(np.float32)),*axes)
        changes[side] = [np.asarray(a).reshape(len(panel),2,-1) for a in (delta,random_edit)]
        norms[side] = np.stack([np.linalg.norm(v,axis=2) for v in changes[side]],axis=-1)
    del donor
    gc.collect()
    host,_ = load_model(paths["host"],strict=True)
    scores = evaluate(host,panel,changes)
    assert all(v.shape==(len(panel),4,4) and np.isfinite(v).all() for v in scores.values())
    result = summarize(scores,panel)
    np.savez_compressed(out/"composition.npz",left_norms=norms["left"],right_norms=norms["right"],**scores)
    (out/"programs.json").write_text(json.dumps(dict(donor=sources,host=panel),separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(dict(models=MODELS,groups=len(panel),map_digest=digest,**result),indent=2)+"\n")


if __name__ == "__main__":
    main()
