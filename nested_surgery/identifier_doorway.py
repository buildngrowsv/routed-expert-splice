"""Constrain propagation of a declaration edit to a single relay token."""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_baseline import SITES, encoded_batches, finish, make_pairs, prefix
from identifier_binding import patch, programs
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

MODES = ("native", "unrestricted", "same_class_relay", "other_class_relay")


def relay_outputs(model, h, modified, sites, end, choices, cut):
    token = mx.arange(h.shape[1])[None, :]
    index = mx.arange(h.shape[0])
    source, same, other = [sites[:,k] for k in range(3)]
    clean = h
    live = [modified, modified, modified]
    gates = [
        ((token == source[:, None]) | (token == relay[:, None]))[..., None]
        for relay in (same,other)
    ]
    mask = create_attention_mask(h,None)
    for block in model.model.layers[cut:cut+7]:
        clean = block(clean,mask)
        live[0] = block(live[0],mask)
        for k, gate in enumerate(gates,1):
            live[k] = mx.where(gate,block(live[k],mask),clean)
    shifts = []
    for k, relay in enumerate((same,other),1):
        arrived = live[k][index,relay]
        shifts.append(np.asarray(mx.linalg.norm((arrived-clean[index,relay]).astype(mx.float32),axis=1)))
        live[k] = patch(clean,relay,arrived)
    outputs = [finish(model,x,end,cut+7,choices) for x in (clean,*live)]
    return outputs,np.column_stack(shifts)


def evaluate(model, rows, axis, cut, *, run_relays=relay_outputs):
    axis = mx.array(axis)
    scores = {m:[] for m in MODES}
    displacements = []
    offset = 0
    for ids, source, end, choices, _ in encoded_batches(rows):
        batch = len(choices)
        example = rows[offset:offset+batch//2]
        offset += len(example)
        positions = mx.array(np.repeat([r["relays"] for r in example],2,axis=0))
        sites = mx.concatenate((source[:,None],positions),axis=1)
        h = prefix(model,ids,cut)
        value = h[mx.arange(batch),source].astype(mx.float32)
        canonical = mx.array(np.arange(batch) & ~2)
        partner = mx.array(np.arange(batch) ^ 1)
        np.testing.assert_array_equal(np.asarray(value),np.asarray(value[canonical]))
        change = ((value[canonical][partner]-value) @ axis) @ axis.T
        modified = patch(h,source,value+change)
        outcomes,shifts = run_relays(model,h,modified,sites,end,choices,cut)
        assert np.isfinite(shifts).all()
        displacements.extend(shifts.reshape(-1,2,2,2))
        for name,margins in zip(MODES,outcomes,strict=True):
            scores[name].extend(margins)
    scores = {m:np.asarray(v).reshape(-1,2,2) for m,v in scores.items()}
    assert all(np.isfinite(v).all() for v in scores.values())
    return scores,np.asarray(displacements)


def summarize(scores,shifts):
    eligible = (scores["native"]>0).all(axis=(1,2))
    n = int(eligible.sum())
    flips = {k:(v[eligible]<0).all(axis=2) for k,v in scores.items() if k!="native"}
    report = dict(decision="unresolved",eligible_groups=n,reversals={},relay_advantage=None)
    if not n:
        return report
    idx = np.random.default_rng(528092).integers(n,size=(5000,n))
    for name, v in flips.items():
        report["reversals"][name] = dict(first=int(v[:,0].sum()),second=int(v[:,1].sum()),
                                         differential=interval(v[:,0].astype(int)-v[:,1],idx))
    report["relay_advantage"] = interval(flips["same_class_relay"][:,0].astype(int)-flips["other_class_relay"][:,0],idx)
    report["median_release_norm"] = np.median(shifts,axis=(0,1,2)).tolist()
    free = report["reversals"]["unrestricted"]
    ready = n>=100 and free["first"]>=.6*n and free["second"]<=.15*n
    gates = {m:v["first"]>=.35*n and v["differential"]["interval"][0]>0
             for m,v in report["reversals"].items() if m.endswith("_relay")}
    report["decision"] = "single_relay" if ready and any(gates.values()) else "unresolved"
    report["qualifying_relays"] = [k for k,v in gates.items() if ready and v]
    return report


def rows_with_relays(tokenizer):
    rows = programs(tokenizer,256,32000)
    for row in rows:
        row["relays"] = [s+1 for s in row["sites"]]
        for ids in row["ids"]:
            assert all(tokenizer.decode([ids[k]]).strip()=="=" for k in row["relays"])
    return rows


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    path = model_files("host")
    tok = Tokenizer.from_file(str(path/"tokenizer.json"))
    rows = rows_with_relays(tok)
    cal = make_pairs([tok],512,0)
    model, _ = load_model(path,strict=True)
    cut = SITES["host"]
    labels = [ord(c)-97 for r in cal for c in r["letters"]]
    axis, _ = bases(gather(model,cal,cut),labels)
    scores, shifts = evaluate(model,rows,axis,cut)
    np.savez_compressed(out/"doorway.npz", relay_release_norms=shifts, **scores)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    report = dict(model=MODELS["host"],cut=cut,relay_blocks=7,groups=len(rows)//2,**summarize(scores,shifts))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
