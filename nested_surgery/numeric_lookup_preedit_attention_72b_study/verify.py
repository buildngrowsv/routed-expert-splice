"""Check equal-size attention controls on the exact archived early key programs."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"
FRESH=HERE.parent/"numeric_lookup_fresh_72b_study"/"results"
STAT=runpy.run_path(str(HERE.parent/"numeric_lookup_72b_study"/"verify.py"))
ARMS=("key_block","syntax_block")


def main():
    r=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "qwen-dictionary-block-original-key-before-edit-72b","5756a583dc54d6dded1a0af60c52598da7b94764","done",0)
    prompt_data=(FRESH/"programs.json").read_bytes()
    early=FRESH/"early.npz"
    assert sha256(prompt_data).hexdigest()==r["source_programs_digest"]
    assert sha256(early.read_bytes()).hexdigest()==r["source_ranks_digest"]
    rows=json.loads(prompt_data)["early"]
    assert len(rows)==768 and r["cut"]==17
    STAT["verify_rows"](rows)
    with np.load(early) as z:
        scores={"unrestricted":{k:z[k] for k in ("native","whole")}}
    with np.load(R/"attention.npz") as z:
        for m in ARMS:
            scores[m]={k:z[f"{m}_{k}"] for k in ("native","whole")}
    native={}
    for arm,raw in scores.items():
        for array in raw.values():
            STAT["RANKS"](rows,array)
        native[arm]=STAT["outcome"](raw["native"])
    assert {m:int(v.sum()) for m,v in native.items()}==r["native"]
    cohort=np.logical_and.reduce(list(native.values()))
    n=int(cohort.sum())
    assert n==r["eligible"]
    switches,preserves={},{}
    for arm,raw in scores.items():
        a=raw["whole"].reshape(256,3,2,4)[cohort]
        switches[arm]=(a[:,:2,:,1]==1).all(axis=(1,2))
        preserves[arm]=int((a[:,2,:,0]==1).all(axis=1).sum())
    total={m:int(v.sum()) for m,v in switches.items()}
    assert total==r["switched"] and preserves==r["constant_preserved"]
    if n:
        for ref in ("unrestricted","syntax_block"):
            expected=r["key_block_minus"][ref]
            actual=STAT["paired"](switches["key_block"],switches[ref],877002)
            np.testing.assert_allclose([actual["mean"],*actual["interval"]],[expected["mean"],*expected["interval"]])
    else:
        assert not r["key_block_minus"]
    qualified=(n>=100 and total["key_block"]>=.8*n and preserves["key_block"]>=.95*n
               and min(s["mean"] for s in r["key_block_minus"].values())>=.15
               and min(s["interval"][0] for s in r["key_block_minus"].values())>0)
    assert r["decision"]==("blocking_preedit_key_access_restores_steering" if qualified else "unresolved")
    pairs={k:(z["native"][...,0]==1).reshape(256,3,2).all(axis=2).sum(axis=0).tolist()
           for k,z in scores.items()}
    print(json.dumps(dict(**r,native_pairs_by_context=pairs,
                          restored_vs_both_controls=int((switches["key_block"] &
                                                         ~switches["syntax_block"] &
                                                         ~switches["unrestricted"]).sum())),indent=2))


if __name__=="__main__":
    main()
