"""Verify the separate frozen 32B host's prompt orders and paired ranks."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"
FRESH=HERE.parent/"numeric_lookup_fresh_72b_study"/"results"
GAP_72=HERE.parent/"numeric_lookup_gap_72b_study"/"results"
CHECK=runpy.run_path(str(HERE.parent/"numeric_lookup_72b_study"/"verify.py"))
ORDERS=("early","late","gap")


def whole(z, cohort):
    x=z.reshape(256,3,2,4)[cohort]
    return (x[:,:2,:,1]==1).all(axis=(1,2)), int((x[:,2,:,0]==1).all(axis=1).sum())


def main():
    report=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "qwen-dictionary-order-second-host-32b","4d0348e77fafbb3a0a6c12015f20cc341197288f","done",0)
    base=(FRESH/"programs.json").read_bytes()
    assert sha256(base).hexdigest()==report["source_programs_digest"]
    prior=json.loads(base)
    panel=json.loads((R/"programs.json").read_text())
    assert set(panel)==set(ORDERS) and all(len(p)==768 for p in panel.values())
    for early,late,gap,before,after in zip(*(panel[k] for k in ORDERS),prior["early"],prior["late"],strict=True):
        assert early["prompts"]==before["prompts"] and late["prompts"]==after["prompts"]
        assert early["answers"]==late["answers"]==gap["answers"]
        assert early["site"]<late["site"]==gap["site"]
        for j,(prompt,digit) in enumerate(zip(gap["prompts"],gap["answers"],strict=True)):
            lines=prompt.splitlines(keepends=True)
            assert lines[3].startswith("    # note ") and "".join(lines[:3]+lines[4:])==late["prompts"][j]
            exec(compile(prompt+digit+"\n","<archived 32B matched gap>","exec"),{})
            cut=gap["site"]
            assert gap["ids"][j][:cut+1]==late["ids"][j][:cut+1]
            assert len(gap["ids"][j])-cut==len(early["ids"][j])-early["site"]==gap["matched_gap"]
    ranks,native,external,external_native={},{},{},{}
    for order in ORDERS:
        with np.load(R/f"{order}.npz") as z:
            ranks[order]=z["whole"]
            native[order]=CHECK["outcome"](z["native"])
            CHECK["RANKS"](panel[order],z["native"])
            CHECK["RANKS"](panel[order],ranks[order])
        base_72=GAP_72 if order=="gap" else FRESH
        with np.load(base_72/f"{order}.npz") as z:
            external_native[order]=CHECK["outcome"](z["native"])
            external[order]=z["whole"]
    assert {k:int(v.sum()) for k,v in native.items()}==report["native"]
    for label,orders,seed in (("paired",ORDERS[:2],732678),("triple",ORDERS,732679)):
        cohort=np.logical_and.reduce([native[k] for k in orders])
        observed=report[label]
        assert int(cohort.sum())==observed["n"]
        outputs={}
        for order in orders:
            answers, preserve=whole(ranks[order],cohort)
            assert int(answers.sum())==observed["switches"][order]
            assert preserve==observed["preserved"][order]
            outputs[order]=answers
        for order in orders[1:]:
            actual=CHECK["paired"](outputs[order],outputs["early"],seed)
            sealed=observed["effects"][order]
            np.testing.assert_allclose([actual["mean"],*actual["interval"]],[sealed["mean"],*sealed["interval"]])
    p=report["paired"]
    n=p["n"]
    effect=p["effects"]["late"]
    passes=(n>=100 and p["switches"]["late"]>=.8*n and p["preserved"]["late"]>=.95*n
            and effect["mean"]>=.15 and effect["interval"][0]>0)
    assert report["decision"]==("late_key_control_replicates_at_32b" if passes else "unresolved")
    both=np.logical_and.reduce([*native.values(),*external_native.values()])
    overlap={name:{k:int(whole(data[k],both)[0].sum()) for k in ORDERS}
             for name,data in (("32b",ranks),("72b",external))}
    print(json.dumps(dict(**report,posthoc_all_six_native_groups=int(both.sum()),
                          posthoc_shared_switches=overlap),indent=2))


if __name__=="__main__":
    main()
