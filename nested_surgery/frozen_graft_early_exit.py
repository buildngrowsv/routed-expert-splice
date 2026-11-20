"""Predict the late coding-port features from a frozen early donor exit."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval, read_logits
from large_graft import HOST_BLOCK, encode, inputs
from large_opener import MODELS, model_files
from learned_output_port import Batch, score
from learned_output_port_decoys import rewrite
from opener_programs import programs

HERE = Path(__file__).resolve().parent
EXIT = 10
BATCH = 12


def read_donor(coder, ids, positions, choices):
    model = coder.model
    h = model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for block in model.layers[:EXIT]:
        h = block(h, mask)
    ix = mx.arange(len(ids)), positions
    early = h[ix]
    for block in model.layers[EXIT:-1]:
        h = block(h, mask)
    final = model.layers[-1]
    r = h + final.self_attn(final.input_layernorm(h), mask)
    expert = final.mlp(final.post_attention_layernorm(r[ix]))
    logits = read_logits(coder, model.norm(r[ix] + expert), choices)
    return np.asarray(early.astype(mx.float32)), np.asarray(expert.astype(mx.float32)), logits


def normalize(values, weights, source):
    return ((values - weights[f"basis_{source}_center"]) @ weights[f"basis_{source}_directions"]
            / weights[f"basis_{source}_scale"])


def fit_maps(inputs_, outputs):
    x = inputs_.astype(np.float64)
    center = x.mean(0)
    x -= center
    gram = x @ x.T
    penalty = .02 * np.diag(gram).mean()
    assert penalty > 0
    gram.flat[::len(gram)+1] += penalty
    targets = {"mapped": outputs, "shuffled": outputs[np.random.default_rng(712530).permutation(len(outputs))]}
    maps = {}
    for name, target in targets.items():
        prior = target.mean(0)
        coefficient = x.T @ np.linalg.solve(gram, target - prior)
        maps[name] = (center.astype(np.float32), coefficient.astype(np.float32), prior)
    return maps


def metrics(logits, predictions, expected):
    n = len(logits["teacher"])
    correct = {k: (z.argmax(-1) == np.arange(3)).all(1) for k,z in logits.items()}
    draws = np.random.default_rng(174609).integers(n,size=(5000,n))
    c = correct["mapped"].astype(int)
    paired = {k: interval(c-correct[k],draws) for k in ("host","shuffled","off")}
    attributable = (correct["mapped"] & correct["teacher"] & ~correct["host"] & ~correct["off"] & ~correct["shuffled"])
    teacher = int(correct["teacher"].sum())
    denominator = np.square(expected).mean()
    errors = {k:float(np.square(v - expected).mean()/denominator) for k,v in predictions.items()}
    passed = (teacher>=64 and c.sum()>=.8*teacher and attributable.sum()>=10
              and all(v["mean"]>=.05 and v["interval"][0]>0 for v in paired.values()))
    return dict(decision="early_exit_preserves_donor_transfer" if passed else "unresolved",
                complete={k:int(v.sum()) for k,v in correct.items()}, paired=paired,
                donor_attributable=int(attributable.sum()), normalized_feature_errors=errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    smoke = ap.parse_args().smoke
    start = time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    original = HERE/"learned_output_port_study/results"
    prior = json.loads((original/"result.json").read_text())
    path = original/"ports.safetensors"
    digest = sha256(path.read_bytes()).hexdigest()
    assert digest == prior["port_sha256"]
    parameters = mx.load(str(path))
    basis = {k:np.asarray(v) for k,v in parameters.items() if k.startswith("basis_")}
    weights = {arm:{k:parameters[f"learner_{arm}_{k}"] for k in ("port","bias")}
               for arm in ("donor","host_feature")}
    paths = {k:model_files(k) for k in MODELS}
    tokens = {k:Tokenizer.from_file(str(p/"tokenizer.json")) for k,p in paths.items()}
    host,_ = load_model(paths["host"], strict=True)
    coder,_ = load_model(paths["donor"], strict=True)
    assert len(coder.model.layers)==28
    train = rewrite(programs(4 if smoke else 192,start=90000),True)
    test = rewrite(programs(4 if smoke else 512,start=91000),True)
    pairs = []
    for j in range(0,len(train),BATCH):
        ids,pos,opts,_ = encode(train[j:j+BATCH],tokens["donor"])
        early,expert,_ = read_donor(coder,mx.array(ids),pos,opts)
        pairs.append((early, normalize(expert,basis,"donor_feature")))
    x,y = map(np.concatenate,zip(*pairs,strict=True))
    fits = fit_maps(x,y)
    logits = {arm:[] for arm in ("mapped","shuffled","teacher","host","off","intact")}
    guessed={k:[] for k in ("mapped","shuffled")}
    reference=[]
    for j in range(0,len(test),BATCH):
        rows = test[j:j+BATCH]
        ids,pos,opts,labels = encode(rows,tokens["host"])
        dids,dpos,dopts,_ = encode(rows,tokens["donor"])
        np.testing.assert_array_equal(dids,ids)
        np.testing.assert_array_equal(np.asarray(dpos),np.asarray(pos))
        np.testing.assert_array_equal(np.asarray(dopts),np.asarray(opts))
        ids = mx.array(ids)
        early,expert,standalone = read_donor(coder,ids,pos,opts)
        if j == 0:
            last = mx.arange(len(ids)),pos
            check = read_logits(coder,coder.model(ids)[last],opts)
            np.testing.assert_array_equal(check.argmax(1),standalone.argmax(1))
        r,u = inputs(host,ids,HOST_BLOCK)
        host_update = host.model.layers[HOST_BLOCK].mlp(u)
        batch = Batch(r+host_update,pos,opts,labels,np.empty(0),np.empty(0),standalone)
        teacher = normalize(expert,basis,"donor_feature")
        reference.append(teacher)
        sources = {"teacher":teacher,"off":np.zeros_like(teacher),
                   "host":normalize(np.asarray(host_update[mx.arange(len(ids)),pos].astype(mx.float32)),
                                    basis,"host_feature")}
        for mode,(center,direction,bias) in fits.items():
            values = (early-center)@direction+bias
            sources[mode]=values
            guessed[mode].append(values)
        for arm,graft in sources.items():
            connector=weights["host_feature" if arm=="host" else "donor"]
            logits[arm].append(np.asarray(score(host,batch,mx.array(graft),connector)))
        logits["intact"].append(standalone)
        if (j//BATCH+1)%24==0:
            print(f"Scored {j+len(rows)}/{len(test)} repair prompts",flush=True)
    scores = {k:np.concatenate(v).reshape(-1,3,4) for k,v in logits.items()}
    inferred = {k:np.concatenate(v) for k,v in guessed.items()}
    target = np.concatenate(reference)
    report = metrics(scores,inferred,target)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    report.update(cut=EXIT,train_groups=len(train)//3,test_groups=len(test)//3,models=MODELS,
                  port_sha256=digest,seconds=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    if smoke:
        assert all(np.isfinite(v).all() for v in (*scores.values(),target))
        report["decision"]="smoke_complete"
    else:
        np.savez_compressed(out/"scores.npz",**scores,teacher_features=target,**{f"features_{k}":v for k,v in inferred.items()})
        np.savez_compressed(out/"maps.npz",**{f"{k}_{i}":v for k,fit in fits.items() for i,v in enumerate(fit)})
        (out/"programs.json").write_text(json.dumps(dict(train=train,test=test),separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
