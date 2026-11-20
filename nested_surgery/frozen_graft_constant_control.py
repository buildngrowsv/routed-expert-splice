"""Test a fixed calibration vector against prompt-dependent early graft features."""
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
from large_graft import HOST_BLOCK, encode, inputs
from large_opener import model_files
from learned_output_port import Batch, score

HERE = Path(__file__).resolve().parent
EARLY = HERE / "frozen_graft_early_exit_study/results"
BATCH = 12


def evaluate(host, tokens, rows, mapped, average, weights):
    donor = mapped.reshape(-1,3,mapped.shape[-1])
    alternatives = {
        "aligned":mapped, "calibration_mean":np.broadcast_to(average,mapped.shape),
        "cyclic":donor[:,[1,2,0]].reshape(mapped.shape),
    }
    scores = {k:[] for k in alternatives}
    for i in range(0,len(rows),BATCH):
        chunk = rows[i:i+BATCH]
        ids,pos,options,labels = encode(chunk,tokens)
        r,u = inputs(host,mx.array(ids),HOST_BLOCK)
        h = r + host.model.layers[HOST_BLOCK].mlp(u)
        unused = np.empty(0)
        batch = Batch(h,pos,options,labels,unused,unused,unused)
        mx.eval(h)
        for k,a in alternatives.items():
            scores[k].append(np.asarray(score(host,batch,mx.array(a[i:i+len(chunk)]),weights)))
        if (i//BATCH+1)%32==0:
            print(f"Scored {i+len(chunk)}/{len(rows)} recipient prompts",flush=True)
    return {k:np.concatenate(v).reshape(-1,3,4) for k,v in scores.items()}


def summarize(scores, archive):
    n=len(scores["aligned"])
    gold=np.arange(3)
    is_correct=lambda values:(values.argmax(-1)==gold).all(1)
    truth={k:is_correct(v) for k,v in (archive | scores).items()}
    aligned,mean=truth["aligned"],truth["calibration_mean"]
    refits=truth["teacher"] & aligned & ~(mean|truth["cyclic"]|truth["host"]|truth["shuffled"]|truth["off"])
    index=np.random.default_rng(731817).integers(n,size=(5000,n))
    relative=interval(aligned.astype(int)-mean,index)
    mean_host=interval(mean.astype(int)-truth["host"],index)
    swapped=(scores["cyclic"].argmax(-1)==[1,2,0]).all(1)
    decision="unresolved"
    if relative["mean"]>=.05 and relative["interval"][0]>0 and refits.sum()>=10:
        decision="prompt_specific_donor_features_help"
    elif relative["interval"][1]<=.02 and mean_host["mean"]>=.05 and mean_host["interval"][0]>0:
        decision="constant_feature_matches_early_graft"
    return dict(decision=decision,complete={k:int(v.sum()) for k,v in truth.items()},
                early_minus_constant=relative,constant_minus_host=mean_host,
                teacher_shared_strict=int(refits.sum()), exact_cyclic_follows=int(swapped.sum()))


def main():
    start=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    base=json.loads((EARLY/"result.json").read_text())
    assert base["decision"]=="unresolved" and base["complete"]["mapped"]==166
    with np.load(EARLY/"scores.npz") as saved:
        mapped=saved["features_mapped"]
        previous={k:saved[k] for k in ("mapped","host","teacher","shuffled","off")}
    with np.load(EARLY/"maps.npz") as saved:
        average=saved["mapped_2"]
    bridge=HERE/"learned_output_port_study/results/ports.safetensors"
    assert sha256(bridge.read_bytes()).hexdigest()==base["port_sha256"]
    saved=mx.load(str(bridge))
    weights={k:saved[f"learner_donor_{k}"] for k in ("port","bias")}
    path=model_files("host")
    host,_=load_model(path,strict=True)
    tokens=Tokenizer.from_file(str(path/"tokenizer.json"))
    prompts=EARLY/"programs.json"
    rows=json.loads(prompts.read_text())["test"]
    scores=evaluate(host,tokens,rows,mapped,average,weights)
    np.testing.assert_array_equal(scores["aligned"].argmax(-1),previous["mapped"].argmax(-1))
    r=summarize(scores,previous)
    r.update(groups=len(rows)//3,seconds=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory(),
             panel_sha256=sha256(prompts.read_bytes()).hexdigest(),port_sha256=base["port_sha256"])
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"constant.npz",**scores)
    (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")


if __name__=="__main__":
    main()
