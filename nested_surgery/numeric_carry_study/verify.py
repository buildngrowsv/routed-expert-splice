"""Execute every arithmetic completion and audit complete greedy token paths."""
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
MODES=("native","both","random","whole")


def verify_programs(rows):
    for r in rows:
        expected=[a+b for a,b in product(r["left"],r["right"])]
        assert list(map(int,r["answers"]))==expected and len(set(expected))==4
        assert min(expected)<10<=max(expected) and set(expected).isdisjoint(r["left"]+r["right"])
        assert [len(t) for t in r["answer_tokens"]]==[len(a) for a in r["answers"]]
        assert [i for i,z in enumerate(zip(*r["ids"],strict=True)) if len(set(z))>1]==r["sites"]
        for variant,prompt in enumerate(r["prompts"]):
            valid=[]
            for candidate in r["answers"]:
                ns={}
                try:
                    exec(compile(prompt+candidate+"\n","<archived carry probe>","exec"),ns)
                except AssertionError:
                    continue
                assert ns["fetch"]()==int(candidate)
                valid.append(int(candidate))
            assert valid==[expected[variant]]


def decode_hits(rows,data,endings):
    output={}
    end_ids=set(endings.tolist())
    for name,v in data.items():
        assert v.shape==(len(rows),4,3,2) and np.issubdtype(v.dtype,np.integer)
        assert (v[...,0]>=0).all() and np.isin(v[...,1],[0,1]).all()
        target=np.array([r["answer_tokens"] for r in rows],dtype=object)
        hits=[]
        for group,records in enumerate(v):
            mask=[]
            for case,trace in enumerate(records):
                choice=case if name=="native" else 3-case
                suffix=target[group,choice]
                n=len(suffix)
                mask.append(all(int(trace[i,0])==digit and trace[i,1]==1 for i,digit in enumerate(suffix))
                            and int(trace[n,0]) in end_ids)
            hits.append(mask)
        output[name]=np.array(hits,bool)
    return output


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="c4062f2af3d71f0167688c21b7bd786c9bc244b3" and receipt["status"]=="done"
    record=json.loads((ROOT/"result.json").read_text())
    screen=json.loads((ROOT/"screen.json").read_text())
    assert [r["id"] for r in screen]==list(range(110000,110128))
    verify_programs(screen)
    with np.load(ROOT/"screen.npz") as z:
        assert set(z.files)=={"native","terminators"}
        baseline=decode_hits(screen,{"native":z["native"]},z["terminators"])
    count=int(baseline["native"].all(1).sum())
    assert record["screen_eligible"]==count and count>=64
    programs=json.loads((ROOT/"programs.json").read_text())
    rows=programs["host"]
    assert [r["id"] for r in rows]==list(range(111000,111512))
    verify_programs(rows)
    for side,start in (("left",112000),("right",113000)):
        source=programs["donor"][side]
        assert [r["id"] for r in source]==list(range(start,start+512))
        for r,host in zip(source,rows,strict=True):
            assert r["source_values"]==host[side] and r["offset"]==0
            stem=re.search(r"amount_\d+",r["prompts"][0]).group()
            assert all(stem not in s for s in host["prompts"])
            for p,a in zip(r["prompts"],r["answers"],strict=True):
                exec(compile(p+a+"\n","<archived carry donor>","exec"),{})
    digest=sha256((HERE.parent/"numeric_transfer_study/results/map.npz").read_bytes()).hexdigest()
    assert digest==record["map_digest"]
    with np.load(ROOT/"carry.npz") as z:
        assert set(z.files)=={*MODES,"left_norms","right_norms","terminators"}
        for m in ("left_norms","right_norms"):
            norms=z[m]
            assert norms.shape==(512,2,2) and np.isfinite(norms).all()
            np.testing.assert_allclose(norms[:,:,0],norms[:,:,1],atol=1e-6,rtol=1e-5)
        hits=decode_hits(rows,{m:z[m] for m in MODES},z["terminators"])
        end_ids=z["terminators"].tolist()
    active=hits["native"].all(1)
    n=int(active.sum())
    together={m:v[active].all(1) for m,v in hits.items()}
    counts={m:int(v.sum()) for m,v in together.items()}
    assert record["eligible"]==n and record["correct"]==counts and n>=100
    widths=np.array([[len(a) for a in r["answers"]] for r in rows])
    for kind,a,b in (("one_to_two",1,2),("two_to_one",2,1)):
        mask=active[:,None]&(widths==a)&(widths[:,::-1]==b)
        audit=dict(prompts=int(mask.sum()),correct={m:int(h[mask].sum()) for m,h in hits.items() if m!="native"})
        assert audit==record["direction_counts"][kind]
    d=together["both"].astype(int)-together["random"]
    idx=np.random.default_rng(814225).integers(n,size=(5000,n))
    contrast=dict(mean=float(d.mean()),interval=np.percentile(d[idx].mean(1),[2.5,97.5]).tolist())
    assert contrast==record["both_minus_random"]
    passed=min(counts[m] for m in ("both","whole"))>=.8*n and d.mean()>=.2 and contrast["interval"][0]>0
    assert record["decision"]==("carry_composes" if passed else "unresolved")
    print(json.dumps(dict(**record,observed_terminator_ids=end_ids),indent=2))


if __name__=="__main__":
    main()
