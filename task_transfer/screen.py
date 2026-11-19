"""Select with discovery only. A single independent confirmation gates model surgery."""
from pathlib import Path
import sys
import json

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"penultimate"))
from features import ARMS, CACHE, load_model, write
from problems import CHOICES, FAMILIES, exact_p, panel, run_patches

OUT = HERE/"results"
WORK = CACHE/"task-transfer"


def encode(rows):
    tok = Tokenizer.from_file(str(CACHE/"host/tokenizer.json"))
    small = Tokenizer.from_file(str(CACHE/"donor/tokenizer.json"))
    family = rows[0]["family"]
    choices = [tok.encode(s,add_special_tokens=False).ids for s in CHOICES[family]]
    assert all(len(x)==1 for x in choices)
    choices = np.array([x[0] for x in choices],dtype=np.int32)
    seq = []
    for row in rows:
        ids = tok.encode(row["prompt"],add_special_tokens=False).ids
        assert ids == small.encode(row["prompt"],add_special_tokens=False).ids
        for token,choice in zip(choices,CHOICES[family]):
            assert tok.encode(row["prompt"]+choice,add_special_tokens=False).ids == ids+[int(token)]
            assert small.encode(choice,add_special_tokens=False).ids == [token]
        seq.append(ids)
    lengths = np.array([len(x) for x in seq],dtype=np.int32)
    ids = np.zeros((len(rows), (max(lengths)+7)//8*8), dtype=np.int32)
    for i,s in enumerate(seq):
        ids[i,:len(s)]=s
    return ids, lengths-1, choices, np.array([r["gold"] for r in rows],dtype=np.int32)


def full_logits(model, dataset):
    ids, pos, choices, _ = dataset
    parts = []
    for i in range(0,len(ids),8):
        xs = ids[i:i+8]
        z = model(mx.array(xs))[mx.arange(len(xs)),mx.array(pos[i:i+8])]
        parts.append(np.array(z[:,mx.array(choices)].astype(mx.float32)))
    return np.concatenate(parts)


def score_panel(models, f, split, rows=None):
    if rows is None:
        rows = panel(f,split)
    dataset = encode(rows)
    gold = dataset[-1]
    pred = {k: full_logits(m,dataset).argmax(1) for k,m in models.items()}
    oracle = np.array([[all(x) for x in run_patches(row)] for row in rows])
    host, donor = (oracle[np.arange(len(rows)),pred[m]] for m in ("host","donor"))
    wins, losses = int((donor & ~host).sum()), int((host & ~donor).sum())
    scores = dict(n=len(rows),host=int(host.sum()),donor=int(donor.sum()),donor_only=wins,host_only=losses,
                  difference=(wins-losses)/len(rows),p=exact_p(wins,losses))
    evidence = [dict(id=r["id"],gold=int(g),host=int(h),donor=int(d))
                for r,g,h,d in zip(rows,gold,pred["host"],pred["donor"])]
    print(split,f,scores,flush=True)
    return scores,evidence


def main():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    models = {n:load_model(n) for n in ("host","donor")}
    discovery, records = {}, []
    for f in FAMILIES:
        discovery[f], results = score_panel(models,f,"discovery")
        records.extend(results)
    chosen = sorted(discovery,key=lambda f:(-discovery[f]["difference"],f))[0]
    confirmation = None
    if discovery[chosen]["difference"] >= .05:
        confirmation, results = score_panel(models,chosen,"confirmation")
        records.extend(results)
    qualified = bool(confirmation and confirmation["difference"]>0 and confirmation["p"]<.05)
    write(OUT/"screen.json",dict(discovery=discovery,candidate=chosen,confirmation=confirmation,qualified=qualified))
    write(OUT/"screen-items.json",records)


if __name__ == "__main__":
    main()
