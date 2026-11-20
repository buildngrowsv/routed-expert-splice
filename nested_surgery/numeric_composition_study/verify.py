"""Derive every intervention target from executed Python and verify the full audit."""
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
ARMS=("native","left","right","both","random","whole")
SOURCE_NAME=re.compile(r"amount_\d+")


def target_matrix(rows, sources):
    expected={m:[] for m in ARMS}
    for row,left,right in zip(rows,sources["left"],sources["right"],strict=True):
        assert left["source_values"]==row["left"] and right["source_values"]==row["right"]
        assert left["offset"]==right["offset"]==0
        donor_stems=[SOURCE_NAME.search(d["prompts"][0]).group() for d in (left,right)]
        assert donor_stems[0]!=donor_stems[1]
        assert all(stem not in p for stem in donor_stems for p in row["prompts"])
        for d in (left,right):
            for prompt,letter,value in zip(d["prompts"],d["answers"],d["source_values"],strict=True):
                ns={}
                exec(compile(prompt+letter+"\n","<archived donor probe>","exec"),ns)
                assert ns["fetch"]()==int(letter)==value
        affected=[j for j,tokens in enumerate(zip(*row["ids"],strict=True)) if len(set(tokens))>1]
        assert affected==row["sites"]
        choices=list(map(int,row["answers"]))
        assert len(set(choices))==4 and set(choices).isdisjoint((*row["left"],*row["right"]))
        values={}
        for pair, prompt in zip(product(row["left"],row["right"]),row["prompts"],strict=True):
            passes=[]
            for candidate in choices:
                ns={}
                try:
                    exec(compile(prompt+str(candidate)+"\n","<archived host probe>","exec"),ns)
                except AssertionError:
                    continue
                assert ns["fetch"]()==candidate
                passes.append(candidate)
            assert len(passes)==1
            values[pair]=choices.index(passes[0])
        targets={m:[] for m in ARMS}
        for a,b in product(row["left"],row["right"]):
            u,v=sum(row["left"])-a,sum(row["right"])-b
            coordinates=((a,b),(u,b),(a,v),(u,v),(u,v),(u,v))
            for mode,operands in zip(ARMS,coordinates,strict=True):
                targets[mode].append(values[operands])
        for m,t in targets.items():
            expected[m].append(t)
    return {m:np.array(v) for m,v in expected.items()}


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["exit"]==0
    assert receipt["revision"]=="b640590b733111878fc3d15497940e4913ab667e"
    result=json.loads((ROOT/"result.json").read_text())
    source_map=HERE.parent/"numeric_transfer_study/results/map.npz"
    assert sha256(source_map.read_bytes()).hexdigest()==result["map_digest"]
    programs=json.loads((ROOT/"programs.json").read_text())
    rows=programs["host"]
    assert [x["id"] for x in rows]==list(range(80000,81024))
    for name,start in (("left",84000),("right",86000)):
        assert [x["id"] for x in programs["donor"][name]]==list(range(start,start+1024))
    targets=target_matrix(rows,programs["donor"])
    with np.load(ROOT/"composition.npz") as z:
        assert set(z.files)=={"left_norms","right_norms",*ARMS}
        for name in ("left_norms","right_norms"):
            norms=z[name]
            assert norms.shape==(1024,2,2) and np.isfinite(norms).all()
            np.testing.assert_allclose(norms[:,:,0],norms[:,:,1],rtol=1e-5,atol=1e-6)
        scores={m:z[m] for m in ARMS}
    assert all(a.shape==(1024,4,4) and np.isfinite(a).all() for a in scores.values())
    margins={}
    for m,z in scores.items():
        gold=targets[m][:,:,None]
        good=np.take_along_axis(z,gold,axis=2)[...,0]
        bad=np.max(np.where(np.arange(4)[None,None,:]==gold,-np.inf,z),axis=2)
        margins[m]=good-bad
    won={m:(v>0).all(1) for m,v in margins.items()}
    eligible=won["native"]
    n=int(eligible.sum())
    cells={m:int(v[eligible].sum()) for m,v in won.items()}
    assert result["eligible"]==n and result["cells"]==cells
    signs=np.array([r["sign"] for r in rows])
    for name,v in (("addition",1),("subtraction",-1)):
        ok=eligible&(signs==v)
        audit=dict(eligible=int(ok.sum()),cells={m:int(x[ok].sum()) for m,x in won.items()})
        assert result["by_operator"][name]==audit
    contrast=None
    if n:
        diff=won["both"][eligible].astype(int)-won["random"][eligible]
        idx=np.random.default_rng(853143).integers(n,size=(5000,n))
        contrast=dict(mean=float(diff.mean()),interval=np.percentile(diff[idx].mean(1),[2.5,97.5]).tolist())
    assert contrast==result["both_minus_random"]
    passes=(n>=100 and min(cells[m] for m in ("left","right","both","whole"))>=.6*n
            and contrast["interval"][0]>.2)
    assert result["decision"]==("composes" if passes else "unresolved")
    native_items=margins["native"]>0
    audit=dict(**result,mean_margin={m:float(x[eligible].mean()) if n else None for m,x in margins.items()},
               native_item_accuracy=float(native_items.mean()),
               native_groups_by_correct_items=np.bincount(native_items.sum(1),minlength=5).tolist())
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
