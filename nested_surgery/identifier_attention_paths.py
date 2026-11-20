"""Separate direct and relayed reads of a projected declaration edit."""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, make_pairs, odds, prefix
from identifier_binding import patch, programs
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

MODES = ("unrestricted", "direct_only", "relay_only")


def masks(width, source, readout):
    positions = mx.arange(width)
    queries = positions[None,:,None]
    keys = positions[None,None,:]
    causal = queries >= keys
    at_source = keys == source[:,None,None]
    self_edge = queries == source[:,None,None]
    final_edge = queries == readout[:,None,None]
    all_rows = mx.broadcast_to(causal,(len(source),width,width))
    direct = causal & (~at_source | self_edge | final_edge)
    relay = causal & (~at_source | ~final_edge)
    return {k:m[:,None,:,:] for k,m in zip(MODES,(all_rows,direct,relay),strict=True)}


def read_masked(model, h, ends, cut, candidates, mask):
    for layer in model.model.layers[cut:]:
        h = layer(h,mask)
    state = model.model.norm(h[mx.arange(len(ends)),ends])
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    return odds(head(state).astype(mx.float32),candidates)


def evaluate(model, rows, axis, cut):
    vectors = mx.array(axis)
    margins = {f"{mode}_{arm}":[] for mode in MODES for arm in ("clean","edited")}
    for i,(ids,source,end,choices,_) in enumerate(encoded_batches(rows)):
        index = mx.arange(len(source))
        canonical = mx.array(np.arange(len(source)) & ~2)
        partner = mx.array(np.arange(len(source)) ^ 1)
        h = prefix(model,ids,cut)
        value = h[index,source].astype(mx.float32)
        np.testing.assert_array_equal(np.array(value),np.array(value[canonical]))
        change = ((value[canonical][partner]-value)@vectors)@vectors.T
        altered = patch(h,source,value+change)
        for mode,mask in masks(ids.shape[1],source,end).items():
            margins[mode+"_clean"].extend(read_masked(model,h,end,cut,choices,mask))
            margins[mode+"_edited"].extend(read_masked(model,altered,end,cut,choices,mask))
        if (i+1)%16==0:
            print(f"Scored {min((i+1)*4,len(rows)//2)}/{len(rows)//2} groups",flush=True)
    return {k:np.array(v).reshape(-1,2,2) for k,v in margins.items()}


def summarize(scores):
    clean = {m:(scores[m+"_clean"]>0).all(axis=(1,2)) for m in MODES}
    matched = np.logical_and.reduce(list(clean.values()))
    n = int(matched.sum())
    report = dict(decision="unresolved",eligible=n,clean_groups={k:int(v.sum()) for k,v in clean.items()})
    if not n:
        return report
    turns = {m:(scores[m+"_edited"][matched]<0).all(axis=2) for m in MODES}
    counts = {m:[int(v) for v in a.sum(0)] for m,a in turns.items()}
    rng = np.random.default_rng(497327)
    samples = rng.integers(n,size=(5000,n))
    left = {m:a[:,0].astype(int) for m,a in turns.items()}
    differences = {m:interval(left[m]-left["unrestricted"],samples) for m in MODES[1:]}
    comparison = interval(left["direct_only"]-left["relay_only"],samples)
    valid = n>=100 and counts["unrestricted"][0]>=.6*n and counts["unrestricted"][1]<=.15*n
    sufficient = [m for m in MODES[1:] if valid and counts[m][0]>=.7*n and differences[m]["interval"][0]>=-.15]
    better = None
    if valid and comparison["interval"][0]>.20:
        better = "direct_only"
    if valid and comparison["interval"][1]<-.20:
        better = "relay_only"
    report.update(decision="routing_resolved" if sufficient or better else "unresolved",
                  reversals_by_query=counts,contrast_from_unrestricted=differences,direct_minus_relay=comparison,
                  sufficient_paths=sufficient,more_effective=better,unrestricted_control_qualified=valid)
    return report


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    path = model_files("host")
    tok = Tokenizer.from_file(str(path/"tokenizer.json"))
    rows = programs(tok,512,90000)
    calibration = make_pairs([tok],512,0)
    model,_ = load_model(path,strict=True)
    cut = SITES["host"]
    labels = [ord(c)-97 for row in calibration for c in row["letters"]]
    axis,_ = bases(gather(model,calibration,cut),labels)
    scores = evaluate(model,rows,axis,cut)
    assert all(x.shape==(512,2,2) and np.isfinite(x).all() for x in scores.values())
    np.savez_compressed(out/"attention-paths.npz",axis=axis,**scores)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    result = dict(model=MODELS["host"],groups=len(rows)//2,cut=cut,**summarize(scores))
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
