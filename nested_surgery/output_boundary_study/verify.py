"""Recover the crossed effects and validate the exact frozen interface identities."""
import ast
from itertools import combinations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE/"results"
MODES = ("donor_native","host_native","host_exact_donor","host_refit_donor")


def main():
    receipt = json.loads((ROOT/"receipt.json").read_text())
    report = json.loads((ROOT/"result.json").read_text())
    source = json.loads((HERE.parent/"recalibrated_graft_study/results/inputs.json").read_text())
    assert receipt["status"]=="done" and receipt["exit"]==0
    assert receipt["revision"]=="52bd5fbce030df2ddb3e1a94ed34445982eb9000"
    assert report["hashes"]["output_archive"]==source["hashes"]["archived"]
    assert report["hashes"]["refit_input"]==source["hashes"]["refit"]
    assert len(report["hashes"]["receiver"])==64
    rows = json.loads((ROOT/"programs.json").read_text())
    assert len({r["prompt"] for r in rows})==768
    for i,r in enumerate(rows):
        j,k = divmod(i,3)
        assert r["gold"]==k and r["id"]==f"opener-{70000+j}-{k}"
        if k==2:
            assert [chars for chars in zip(*(a["prompt"] for a in rows[i-2:i+1]),strict=True)
                    if len(set(chars))>1]==[("(","[","{")]
        p,s = r["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        possibles = []
        for l,c in enumerate((")","]","}",":")):
            try:
                ast.parse(p+c+s)
            except SyntaxError:
                continue
            possibles.append(l)
        assert possibles==[k]
    idx = np.random.default_rng(745029).choice(256,(5000,256))
    pairs,complete = {},{}
    with np.load(ROOT/"crossed-scores.npz") as data:
        for mode in MODES:
            x = data[mode]
            assert x.shape==(256,3,3,4) and np.isfinite(x).all()
            predictions = np.stack([x[:,k,k,:].argmax(1) for k in range(3)],axis=1)
            complete[mode] = int((predictions==np.arange(3)).all(1).sum())
            pairs[mode] = np.column_stack([(x[:,:,a,a]-x[:,:,b,a]-x[:,:,a,b]+x[:,:,b,b]).mean(1)
                                           for a,b in combinations(range(3),2)])
    np.testing.assert_equal(complete,report["complete"])
    means = {k:v.mean(1) for k,v in pairs.items()}
    def check(v,s):
        p = float(v.mean())
        ci = np.quantile(v[idx].mean(1),[.025,.975]).tolist()
        np.testing.assert_allclose([p,*ci],[s["mean"],*s["interval"]],atol=1e-6)
        return dict(mean=p,interval=ci)
    for k,v in pairs.items():
        np.testing.assert_allclose(v.mean(0),report["pair_means"][k],atol=1e-6)
    effects = {k:check(v,report["effects"][k]) for k,v in means.items()}
    halves = {k:check(means[k]-.5*means["donor_native"],s) for k,s in report["host_minus_half_native_donor"].items()}
    native = effects["donor_native"]["interval"][0]
    exact,refit = [halves[k]["interval"] for k in ("host_exact_donor","host_refit_donor")]
    status = "unresolved"
    if native>0:
        if exact[1]<0:
            status="output_below_margin"
        elif exact[0]>0 and refit[1]<0:
            status="input_gap_in_host"
        elif refit[0]>0:
            status="refitted_local_signal"
    assert status==report["decision"]
    print(json.dumps(dict(effects=effects,host_minus_half_native_donor=halves,complete=complete,decision=status),indent=2))


if __name__=="__main__":
    main()
