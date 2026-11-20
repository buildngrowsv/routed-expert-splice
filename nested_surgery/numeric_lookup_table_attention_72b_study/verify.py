"""Validate the dictionary-only mask's restricted spans and paired outcomes."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"
OLD=runpy.run_path(str(HERE.parent/"numeric_lookup_preedit_attention_72b_study"/"verify.py"))
FRESH=OLD["FRESH"]
STAT=OLD["STAT"]


def main():
    r=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "qwen-dictionary-table-only-key-block-72b","d29e00dde9be0f7ea6d08a5e427c67bfd9e834ce","done",0)
    assert r["table_only"] and r["cut"]==17
    prompts=(FRESH/"programs.json").read_bytes()
    raw=FRESH/"early.npz"
    assert sha256(prompts).hexdigest()==r["source_programs_digest"]
    assert sha256(raw.read_bytes()).hexdigest()==r["source_ranks_digest"]
    rows=json.loads(prompts)["early"]
    STAT["verify_rows"](rows)
    with np.load(raw) as f:
        panels={"unrestricted":{k:f[k] for k in ("native","whole")}}
    with np.load(R/"attention.npz") as f:
        spans=f["table_spans"]
        for arm in OLD["ARMS"]:
            panels[arm]={k:f[f"{arm}_{k}"] for k in ("native","whole")}
    assert spans.shape==(1536,2)
    # The source-tokenizer smoke found the same half-open span in all 1,536 prompts.
    np.testing.assert_array_equal(spans,np.tile((16,31),(len(spans),1)))
    for row,points in zip(rows,spans.reshape(768,2,2),strict=True):
        for p,ids,(left,right) in zip(row["prompts"],row["ids"],points,strict=True):
            lines=p.splitlines()
            assert lines[2].startswith("    table =") and lines[3].startswith("    return table[")
            assert row["site"]<left<right<len(ids)
    native={}
    for arm,scores in panels.items():
        for x in scores.values():
            STAT["RANKS"](rows,x)
        native[arm]=STAT["outcome"](scores["native"])
    assert {k:int(v.sum()) for k,v in native.items()}==r["native"]
    common=np.logical_and.reduce(list(native.values()))
    n=int(common.sum())
    assert n==r["eligible"]
    switch,keep={},{}
    for arm,scores in panels.items():
        x=scores["whole"].reshape(256,3,2,4)[common]
        switch[arm]=(x[:,:2,:,1]==1).all(axis=(1,2))
        keep[arm]=int((x[:,2,:,0]==1).all(axis=1).sum())
    counts={k:int(v.sum()) for k,v in switch.items()}
    assert counts==r["switched"] and keep==r["constant_preserved"]
    for mode in ("unrestricted","syntax_block"):
        p=STAT["paired"](switch["key_block"],switch[mode],877002)
        old=r["key_block_minus"][mode]
        np.testing.assert_allclose([p["mean"],*p["interval"]],[old["mean"],*old["interval"]])
    passed=(n>=100 and counts["key_block"]>=.8*n and keep["key_block"]>=.95*n
            and min(a["mean"] for a in r["key_block_minus"].values())>=.15
            and min(a["interval"][0] for a in r["key_block_minus"].values())>0)
    assert r["decision"]==("blocking_preedit_key_access_restores_steering" if passed else "unresolved")
    print(json.dumps(dict(**r,table_token_span=spans[0].tolist(),minimum_switches=int(np.ceil(.8*n)),
                          only_key_block=int((switch["key_block"]&~switch["syntax_block"]&~switch["unrestricted"]).sum())),
                     indent=2))


if __name__=="__main__":
    main()
