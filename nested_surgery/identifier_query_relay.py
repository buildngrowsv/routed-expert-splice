"""Compare final-token capture with enforced direct source-to-query propagation."""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES, finish, make_pairs
from identifier_binding import patch, programs
from identifier_doorway import evaluate, summarize
from identifier_subspace import bases
from identifier_transfer import gather
from large_opener import MODELS, model_files

LABELS = {"same_class_relay":"query_capture","other_class_relay":"query_direct"}


def query_outputs(model,h,modified,sites,end,choices,cut):
    token = mx.arange(h.shape[1])[None,:]
    index = mx.arange(h.shape[0])
    gate = ((token==sites[:,0,None])|(token==end[:,None]))[...,None]
    clean, full, direct = h, modified, modified
    mask = create_attention_mask(h,None)
    for block in model.model.layers[cut:cut+7]:
        clean = block(clean,mask)
        full = block(full,mask)
        direct = mx.where(gate,block(direct,mask),clean)
    capture = (full[index,end],direct[index,end])
    states = (clean, full, *(patch(clean,end,value) for value in capture))
    norms = np.column_stack([np.asarray(mx.linalg.norm(value.astype(mx.float32)-clean[index,end].astype(mx.float32),axis=1))
                             for value in capture])
    return [finish(model,x,end,cut+7,choices) for x in states],norms


def rows_with_query(tok):
    rows = programs(tok,256,44500)
    for row in rows:
        a,b = row["ids"]
        assert len(a)==len(b) and len(a)-1>row["sites"][-1]
        row["relays"] = [len(a)-1]*2
    return rows


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    path = model_files("host")
    tok = Tokenizer.from_file(str(path/"tokenizer.json"))
    cal = make_pairs([tok],512,0)
    rows = rows_with_query(tok)
    model,_ = load_model(path,strict=True)
    cut = SITES["host"]
    labels = [ord(c)-97 for row in cal for c in row["letters"]]
    axis,_ = bases(gather(model,cal,cut),labels)
    scores, norms = evaluate(model,rows,axis,cut,run_relays=query_outputs)
    summary = summarize(scores,norms)
    summary["reversals"] = {LABELS.get(k,k):v for k,v in summary["reversals"].items()}
    summary["qualifying_relays"] = [LABELS[k] for k in summary.get("qualifying_relays",[])]
    summary["capture_minus_direct"] = summary.pop("relay_advantage")
    np.savez_compressed(out/"query-relay.npz",relay_release_norms=norms,**{LABELS.get(k,k):v for k,v in scores.items()})
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    result = dict(model=MODELS["host"],cut=cut,relay_blocks=7,groups=len(rows)//2,**summary)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
