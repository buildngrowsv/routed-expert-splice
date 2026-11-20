"""Independently compose frozen or adapted modules in activation space; verify paired ledgers."""
from collections import Counter
from pathlib import Path
import ast
import sys
import json
import numpy as np
from experiment import (R,ROOT,SEEDS,checkpoint,encode,load_model,mx,new_panel,rows_for,solve,
                        trained_path,write,PATCHES,archived_hashes,full_logits)
from features import create_attention_mask

def key_from_ast(row):
    a,b=row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
    parsed=ast.parse(a+"    "+b)
    base=ast.literal_eval(parsed.body[0].body[0].value)
    increments=[n for n in ast.walk(parsed.body[0]) if isinstance(n,ast.AugAssign)]
    assert len(increments)==1
    bonus=ast.literal_eval(increments[0].value)
    vals=[ast.literal_eval(node.test.comparators[0]) for node in parsed.body[1:]]
    return tuple(sorted((v-base)//bonus for v in vals))

def independent_shortcut(train,rows):
    table=Counter((key_from_ast(r),r["gold"]) for r in train)
    keys={key for key,_ in table}
    def predict(row):
        v=key_from_ast(row)
        distances={key:sum(abs(a-b) for a,b in zip(v,key)) for key in keys}
        close=[key for key in keys if distances[key]==min(distances.values())]
        counts=[sum(table[(key,d)] for key in close) for d in range(4)]
        return counts
    return np.array([predict(r) for r in rows])

def composed_scores(base,adapted,data):
    ids,position,choices,_=data
    raw=base.model.layers[-1];new=adapted.model.layers[-1]
    result={k:[] for k in ("full","expert","attention","donor_base")}
    for i in range(0,len(ids),8):
        x=base.model.embed_tokens(mx.array(ids[i:i+8]))
        mask=create_attention_mask(x,None)
        for block in base.model.layers[:-1]:
            x=block(x,mask)
        for attn,mlps in ((raw,((raw.mlp,"donor_base"),(new.mlp,"expert"))),
                          (new,((raw.mlp,"attention"),(new.mlp,"full")))):
            h=x+attn.self_attn(attn.input_layernorm(x),mask)
            u=attn.post_attention_layernorm(h)
            for mlp,name in mlps:
                output=base.model.norm(h+mlp(u))
                # Same tied output head; select the scoring position before projecting.
                at=output[mx.arange(len(output)),mx.array(position[i:i+8])]
                scores=base.model.embed_tokens.as_linear(at)[:,mx.array(choices)]
                result[name].append(np.array(scores.astype(mx.float32)))
    return {k:np.concatenate(parts) for k,parts in result.items()}

def main():
    from hashlib import sha256
    archived_hashes()
    selection=json.loads((R/"selection.json").read_text())
    legacy_train=rows_for("train")
    prior_test=json.loads((ROOT/"specialist"/"results"/"test-problems.json").read_text())
    assert rows_for("test")==prior_test
    seen={r["family"] for r in legacy_train+prior_test+rows_for("calibration")+rows_for("qualification")}
    audit={}
    for split in ("qualification","confirmation"):
        rows=new_panel(split)
        assert rows==json.loads((R/f"{split}-problems.json").read_text())
        fam={r["family"] for r in rows}
        assert len(fam)==16 and not fam&seen
        assert all(sum(int(c)>=4 for c in f)==1 for f in fam)
        seen.update(fam)
        assert all(solve(r)==r["gold"] for r in rows)
        data=encode(rows,PATCHES)
        ledger=json.loads((R/f"{split}-switches.json").read_text())
        differences={}
        with np.load(R/f"{split}.npz") as saved:
            np.testing.assert_array_equal(independent_shortcut(legacy_train,rows),saved["shortcut"])
            np.testing.assert_array_equal(data[-1],saved["gold"])
            raw,adapted=load_model("donor"),load_model("donor")
            previous,host=load_model("donor"),load_model("host")
            np.testing.assert_array_equal(full_logits(host,data),saved["host_base"])
            for seed in SEEDS:
                assert sha256(trained_path(seed).read_bytes()).hexdigest()==selection[str(seed)]["sha256"]
                adapted.model.layers[-1].load_weights(str(trained_path(seed)))
                replays=composed_scores(raw,adapted,data)
                for name,z in replays.items():
                    original=saved[name if name=="donor_base" else f"{name}_{seed}"]
                    differences[f"{name}_{seed}"]=dict(top_one_mismatches=int((z.argmax(1)!=original.argmax(1)).sum()),
                                max_absolute_logit_error=float(np.abs(z-original).max()))
                for instance,module,path,name in ((previous,previous.model.layers[-1].mlp,checkpoint("donor",seed),"previous_expert"),
                                                  (host,host.model.layers[-1],checkpoint("host",seed),"host")):
                    module.load_weights(str(path))
                    np.testing.assert_array_equal(full_logits(instance,data),saved[f"{name}_{seed}"])
                h=saved[f"full_{seed}"].argmax(1)==data[-1]
                for k,record in ledger[str(seed)].items():
                    other=saved[f"{k}_{seed}"].argmax(1)==data[-1]
                    assert record["full_only"]==[r["id"] for r,keep in zip(rows,h&~other) if keep]
                    assert record["other_only"]==[r["id"] for r,keep in zip(rows,other&~h) if keep]
        audit[split]=dict(verified_families=len(fam),verified_repairs=len(rows),verified_quartets=len(rows)//4,
                          shortcut_reproduced=True,ledger_reproduced=True,archived_logits_exact=True,
                          composed_replay=differences)
    write(R/"replay.json",audit)
    for key,a in audit.items():
        print(key,"max logit diff",max(z["max_absolute_logit_error"] for z in a["composed_replay"].values()),
              "top-one mismatches",sum(z["top_one_mismatches"] for z in a["composed_replay"].values()))
if __name__=="__main__":main()

