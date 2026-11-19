"""Actual model forwards for held-out repairs, including withdrawal without retraining."""
import json
import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
from delimiters import delimiter_panel, solved
from screen import ARMS, OUT, WORK, encode, full_logits, load_model, write
from surgery import graft_scores
from features import modules
from forced import SEEDS


def main():
    assert len(json.loads((OUT/"selection.json").read_text()))==len(SEEDS)*len(ARMS)
    rows=delimiter_panel("test")
    data, plain = encode(rows),encode(delimiter_panel("test",plain=True))
    host,donor,general = [load_model(n) for n in ("host","donor","general")]
    arms = modules(host,donor,general)
    block = host.model.layers[-1]
    initial = tree_map(lambda x:x,block.parameters())
    logits = {"donor":full_logits(donor,data)[None,:], "plain_donor":full_logits(donor,plain)[None,:],
              "plain_host":full_logits(host,plain)[None,:]}
    for arm in ARMS:
        block.update(initial)
        logits["frozen_"+arm] = graft_scores(host,arms[arm],data)[None,:]
        adapted = []
        switches = {"coder":("knockout","shuffled","general"),"shuffled":("coder",),"general":("coder",)}.get(arm,())
        cross = {s:[] for s in switches}
        for seed in SEEDS:
            block.load_weights(str(WORK/f"task-{arm}-{seed}.safetensors"))
            adapted.append(graft_scores(host,arms[arm],data))
            for s in switches:
                cross[s].append(graft_scores(host,arms[s],data))
        logits["trained_"+arm] = np.stack(adapted)
        for s,outputs in cross.items():
            logits[arm+"_on_"+s] = np.stack(outputs)
        print("Held-out repairs scored",arm,flush=True)
    gold = data[-1]
    np.savez_compressed(OUT/"logits.npz",**logits,gold=gold,patch_token_ids=data[2])
    for r in rows:
        r["candidate_tests"]=solved(r)
    write(OUT/"test-problems.json",rows)


if __name__=="__main__":
    main()
