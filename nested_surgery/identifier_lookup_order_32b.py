"""Test whole-state dictionary placement and the matched gap in a second frozen host."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from identifier_lookup_gap_72b import FRESH, REVISION, exact_gap
from identifier_lookup_order_72b import evaluate
from identifier_numeric_carry_32b import CUT, HOST_32, verified_result
from identifier_numeric_route_72b import good_groups
from large_opener import checkpoint_files


def encode(tokenizer, rows):
    result = []
    for row in rows:
        prompts = row["prompts"]
        ids = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
        site, = [i for i, (x,y) in enumerate(zip(*ids,strict=True)) if x!=y]
        full = [tokenizer.encode(p+d,add_special_tokens=False).ids
                for p,d in zip(prompts,row["answers"],strict=True)]
        assert [t[:-1] for t in full] == ids
        assert [tokenizer.decode([t[site]]).strip() for t in ids] == list(map(str,row["selector"]))
        result.append(dict(row,ids=ids,site=site,choices=[x[-1] for x in full]))
    return result


def comparison(scores, orders, seed):
    native = np.logical_and.reduce([good_groups(scores[k]["native"]) for k in orders])
    n = int(native.sum())
    outcomes, preserved = {}, {}
    for order in orders:
        a = scores[order]["whole"].reshape(-1,3,2,4)[native]
        outcomes[order] = (a[:,:2,:,1]==1).all(axis=(1,2))
        preserved[order] = int((a[:,2,:,0]==1).all(axis=1).sum())
    effects = {}
    if n:
        draws = np.random.default_rng(seed).integers(n,size=(5000,n))
        b = outcomes[orders[0]].astype(int)
        effects = {order:interval(outcomes[order].astype(int)-b,draws) for order in orders[1:]}
    return dict(n=n,switches={k:int(v.sum()) for k,v in outcomes.items()},preserved=preserved,effects=effects)


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    root = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/FRESH
    parent = verified_result(root,REVISION,"late_key_donor_transfer")
    payload = (root/"programs.json").read_bytes()
    assert sha256(payload).hexdigest() == parent["programs_digest"]
    prompts = json.loads(payload)
    host_path = checkpoint_files(*HOST_32)
    tokenizer = Tokenizer.from_file(str(host_path/"tokenizer.json"))
    rows = {k:encode(tokenizer,prompts[k]) for k in ("early","late")}
    rows["gap"] = exact_gap(tokenizer,rows["early"],rows["late"])
    model,_ = load_model(host_path,strict=True)
    scores = {}
    for order, panel in rows.items():
        print(f"Starting {order} order",flush=True)
        scores[order] = evaluate(model,panel,cut=CUT)
        np.savez_compressed(out/f"{order}.npz",**scores[order])
    paired = comparison(scores,("early","late"),732678)
    triple = comparison(scores,("early","late","gap"),732679)
    n = paired["n"]
    preserves = paired["preserved"]["late"]
    gain = paired["effects"].get("late")
    positive = (n>=100 and paired["switches"]["late"]>=.8*n and preserves>=.95*n
                and gain["mean"]>=.15 and gain["interval"][0]>0)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    result = dict(decision="late_key_control_replicates_at_32b" if positive else "unresolved",
                  recipient=HOST_32,cut=CUT,source_programs_digest=parent["programs_digest"],
                  native={m:int(good_groups(a["native"]).sum()) for m,a in scores.items()},
                  paired=paired,triple=triple,runtime_s=time.perf_counter()-start,
                  peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
