"""Verify the exact key-to-answer token distance and the shared-cohort intervention."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"
FRESH=HERE.parent/"numeric_lookup_fresh_72b_study"/"results"
STAT=runpy.run_path(str(HERE.parent/"numeric_lookup_72b_study/verify.py"))


def main():
    report=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "qwen-dictionary-matched-token-gap-72b","c2367e99f66afcd15d61dc053660401bf3062ce6","done",0)
    for key,path in (("map_digest",HERE.parent/"numeric_absolute_study/results/map.npz"),
                     ("alignment_digest",HERE.parent/"numeric_width_study/72b/alignment.npz"),
                     ("original_programs_digest",FRESH/"programs.json")):
        assert sha256(path.read_bytes()).hexdigest()==report[key]
    rows=json.loads((FRESH/"programs.json").read_text())
    rows["gap"]=json.loads((R/"programs.json").read_text())
    widths=[]
    for a,b,c in zip(rows["early"],rows["late"],rows["gap"],strict=True):
        assert a["id"]==b["id"]==c["id"] and a["choices"]==b["choices"]==c["choices"]
        assert c["site"]==b["site"]
        for j, (text,answer) in enumerate(zip(c["prompts"],c["answers"],strict=True)):
            lines=text.splitlines(keepends=True)
            assert lines[3].startswith("    # note") and "".join(lines[:3]+lines[4:])==b["prompts"][j]
            exec(compile(text+answer+"\n","<archived matched gap>","exec"),{})
            site=c["site"]
            assert c["ids"][j][:site+1]==b["ids"][j][:site+1]
            distance=len(c["ids"][j])-site
            assert distance==len(a["ids"][j])-a["site"]==c["matched_gap"]
            widths.append(distance)
        assert [i for i,(x,y) in enumerate(zip(*c["ids"],strict=True)) if x!=y]==[c["site"]]
    ranks, native={}, {}
    for name in ("early","late","gap"):
        file=(R if name=="gap" else FRESH)/f"{name}.npz"
        if name!="gap":
            assert sha256(file.read_bytes()).hexdigest()==report["original_ranks_digests"][name]
        with np.load(file) as f:
            norms=f["norms"]
            assert norms.shape==(1536,2) and np.isfinite(norms).all()
            np.testing.assert_allclose(norms[:,0],norms[:,1],rtol=1e-5,atol=1e-6)
            ranks[name]={k:f[k] for k in ("native","donor","whole","orthogonal")}
        for data in ranks[name].values():
            STAT["RANKS"](rows[name],data)
        native[name]=STAT["outcome"](ranks[name]["native"])
    assert {k:int(v.sum()) for k,v in native.items()}==report["native"]
    cohort=np.logical_and.reduce(list(native.values()))
    n=int(cohort.sum())
    assert n==report["shared"]
    switched, preserved={}, {}
    for layout, ar in ranks.items():
        for mode in ("donor","whole","orthogonal"):
            a=ar[mode].reshape(256,3,2,4)[cohort]
            key=f"{layout}_{mode}"
            switched[key]=(a[:,:2,:,1]==1).all(axis=(1,2))
            preserved[key]=int((a[:,2,:,0]==1).all(axis=1).sum())
    counts={k:int(v.sum()) for k,v in switched.items()}
    assert counts==report["switches"] and preserved==report["preserved"]
    for label,baseline in (("gap_minus_early","early_donor"),("gap_minus_late","late_donor"),
                           ("gap_minus_orthogonal","gap_orthogonal")):
        actual=STAT["paired"](switched["gap_donor"],switched[baseline],764554)
        expected=report["paired"][label]
        np.testing.assert_allclose([actual["mean"],*actual["interval"]],[expected["mean"],*expected["interval"]])
    gain, delta, ctrl=(report["paired"][x] for x in ("gap_minus_early","gap_minus_late","gap_minus_orthogonal"))
    state="unresolved"
    if n>=100 and preserved["gap_donor"]>=.95*n:
        if (min(counts[k] for k in ("gap_donor","gap_whole"))>=.8*n and gain["mean"]>=.15
                and gain["interval"][0]>0 and delta["interval"][0]>=-.10 and ctrl["mean"]>=.20 and ctrl["interval"][0]>0):
            state="late_key_gain_survives_matched_gap"
        elif delta["mean"]<=-.15 and delta["interval"][1]<0:
            state="neutral_gap_reduces_late_key_control"
    assert report["decision"]==state
    print(json.dumps(dict(**report,exact_gaps=sorted(set(widths)),exact_late_prefix=True),indent=2))


if __name__=="__main__":
    main()
