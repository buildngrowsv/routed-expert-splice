"""Audit the suffix-neutral repairs without importing the training model or generator."""
import ast
import json
from pathlib import Path
import sys
import numpy as np

def main(folder):
    rows=json.loads((folder/"programs.json").read_text())
    report=json.loads((folder/"result.json").read_text())
    logits=np.load(folder/"logits.npz")
    prompts=[r["prompt"] for r in rows]
    assert len(prompts)==len(set(prompts))==768
    tails=set()
    recovered=[]
    for i in range(0,len(rows),3):
        members=prompts[i:i+3]
        changed=[j for j,chars in enumerate(zip(*members)) if len(set(chars))>1]
        assert len(set(map(len,members)))==1 and len(changed)==1
        assert [s[changed[0]] for s in members]==list("([{")
        for s in members:
            h,t=s.removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
            tails.add(t)
            candidates=[]
            for j,c in enumerate((")","]","}",":")):
                try:tree=ast.parse(h+c+t)
                except SyntaxError:continue
                env={"__builtins__":{"sum":sum}}
                exec(compile(tree,"<released delimiter exercise>","exec"),env)
                assert all(env["transform"](x)==x for x in (-913,0,42,719))
                candidates.append(j)
            assert len(candidates)==1
            recovered.append(candidates[0])
    assert len(tails)==1
    np.testing.assert_array_equal(recovered,[r["gold"] for r in rows])
    np.testing.assert_array_equal(recovered,logits["gold"])
    ok={k:logits[k].argmax(1)==recovered for k in logits.files if k!="gold"}
    trips={k:v.reshape(-1,3).all(1) for k,v in ok.items()}
    for k in ok:
        assert int(ok[k].sum())==report["scores"][k] and int(trips[k].sum())==report["triplets"][k]
    strict=trips["coder"]&~trips["native"]&~trips["pre-native"]&~trips["off"]
    assert np.flatnonzero(strict).tolist()==report["strict_disabled_triplets"]
    strict_items=ok["coder"]&~ok["native"]&~ok["pre-native"]&~ok["off"]
    assert [r["id"] for r,b in zip(rows,strict_items) if b]==report["strict_disabled_item_ids"]
    delta=trips["coder"].astype(float)-trips["native"]
    rng=np.random.default_rng(7812);draw=rng.choice(len(delta),size=(5000,len(delta)))
    np.testing.assert_allclose(np.quantile(delta[draw].mean(1),[.025,.975]),report["interval"])
    losses=int((trips["coder"]&~trips["off"]).sum())
    rescued=int((~trips["coder"]&trips["off"]).sum())
    print(json.dumps(dict(programs=len(rows),changes_per_triplet=1,distinct_suffixes=len(tails),
        parser_solved=len(rows),disabled_triplet_losses=losses,disabled_triplet_rescues=rescued,
        strict_triplets=int(strict.sum()),strict_items=int(strict_items.sum()),all_solve_counts_reproduced=True),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]))
