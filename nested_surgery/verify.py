"""Rebuild executable solve outcomes and item-level donor effects from released evidence."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def main(folder):
    rows=json.loads((folder/"programs.json").read_text())
    z=np.load(folder/"logits.npz")
    expected=[]
    for r in rows:
        prompt=r["prompt"]
        assert hashlib.sha256(prompt.encode()).hexdigest()==r["hash"]
        head,tail=prompt.removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        winners=[]
        for i,patch in enumerate((")","]","}",":")):
            try:parsed=ast.parse(head+patch+tail)
            except SyntaxError:continue
            scope={"__builtins__":{"sum":sum}}
            exec(compile(parsed,"<released generated template>","exec"),scope)
            if all(scope["transform"](x)==y for x,y in r["hidden"]):winners.append(i)
        assert winners==[r["gold"]]
        expected.append(winners[0])
    np.testing.assert_array_equal(expected,z["gold"])
    ok={k:z[k].argmax(1)==expected for k in z.files if k!="gold"}
    counts={k:int(v.sum()) for k,v in ok.items()}
    record=json.loads((folder/"result.json").read_text())
    if "native" not in ok:
        assert counts["host"]==record["host"] and counts["donor"]==record["donor"]
        return dict(solves=counts,verified_programs=len(rows))
    assert counts==record["solves"]
    base=ok["pre-native"];adapted=ok["coder"]
    new=adapted&~base;strict=new&~ok["native"]
    def ids(mask):return [r["id"] for r,v in zip(rows,mask) if v]
    losses=ids(new&~ok["off"]);strict_losses=ids(strict&~ok["off"])
    assert len(losses)==record["new_lost_off"] and strict_losses==record["strict_lost_ids"]
    return dict(solves=counts,verified_programs=len(rows),new_count=int(new.sum()),strict_new=int(strict.sum()),
        donor_dependent_new_ids=losses,strict_dependent_ids=strict_losses,
        disabled_losses=int((adapted&~ok["off"]).sum()),disabled_rescues=int((~adapted&ok["off"]).sum()),
        general_vs_native_gains=int((ok["general"]&~ok["native"]).sum()),general_vs_native_losses=int((~ok["general"]&ok["native"]).sum()))
if __name__=="__main__":
    roots=list(map(Path,sys.argv[1:]))
    output={k:main(v) for k,v in zip(("qualification","fit"),roots)}
    q,t=[json.loads((v/"programs.json").read_text()) for v in roots]
    suffix=lambda r:r["prompt"].split("<|fim_suffix|>")[1]
    mapping={suffix(r):r["gold"] for r in q}
    assert all(mapping[suffix(r)]==r["gold"] for r in q)
    output["suffix_lookup"]=dict(entries=len(mapping),fit_panel_correct=sum(mapping[suffix(r)]==r["gold"] for r in t))
    print(json.dumps(output,indent=2))
