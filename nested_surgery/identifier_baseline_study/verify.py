"""Recount decodability and the causal tests on the aligned-token Qwen panel."""
import ast
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"


def main():
    data = json.loads((ROOT/"result.json").read_text())
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "78c3181ae1f7dc2c044ad8b8a14ef73c46ae2731"
    rows = json.loads((ROOT/"test-programs.json").read_text())
    assert [row["id"] for row in rows] == list(range(4096,4608))
    for row in rows:
        a,b = row["ids"]
        assert [i for i,(p,q) in enumerate(zip(a,b,strict=True)) if p!=q] == [row["site"]]
        first,second = row["prompts"]
        assert [chars for chars in zip(first,second,strict=True) if chars[0]!=chars[1]] == [tuple(row["letters"])]
        assert len(set(row["choices"]))==2
        for x,c in zip(row["prompts"],row["letters"],strict=True):
            ast.parse(x+c+"\n")
    checked = {}
    for key,(width,cut) in (("donor",(3584,6)),("host",(5120,10))):
        with np.load(ROOT/(key+"-margins.npz")) as archive:
            scores = dict(archive)
        assert all(v.shape==(512,2) and np.isfinite(v).all() for v in scores.values())
        np.testing.assert_array_equal(scores["whole_state"],-scores["native"][:,::-1])
        base = (scores["native"]>0).all(1)
        n = int(base.sum())
        swaps = {mode:int((base & (scores[mode]<0).all(1)).sum())
                 for mode in ("last","adjacent","whole_site","whole_state")}
        probes = {mode:int((scores["probe_"+mode]>0).all(1).sum()) for mode in ("last","adjacent")}
        reproduced = (n>=100 and min(probes.values())>=410 and swaps["whole_site"]>=.2*n
                      and max(swaps[k] for k in probes)<.05*n)
        actual = dict(width=width,lane=width//8,cut=cut,eligible=n,switches=swaps,decoded=probes,reproduces=reproduced)
        assert actual==data["arms"][key]
        idx = np.random.default_rng(25704).choice(512,(5000,512))
        shifts = {}
        for mode in ("last","adjacent","whole_site"):
            delta = (scores[mode]-scores["native"]).mean(1)
            shifts[mode] = dict(
                delta=float(delta.mean()), interval=np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),
                original_correct_member_flips=int(((scores["native"]>0)&(scores[mode]<0)).sum()),
            )
        checked[key] = {**actual,"exploratory_log_odds_shifts":shifts}
    print(json.dumps(checked,indent=2))


if __name__=="__main__":
    main()
