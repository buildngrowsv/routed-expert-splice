"""Recover the bracket signal and the selection of the ordinary calibration."""
import ast
from itertools import combinations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results"


def main():
    result = json.loads((DATA / "result.json").read_text())
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["revision"] == "e178e4b01c8890a1738d89da5d4e1b79bff56097"
    rows = json.loads((DATA / "programs.json").read_text())
    ids = json.loads((DATA / "calibration-ids.json").read_text())
    assert ids == [f"opener-{i}-{j}" for i in range(45000,45512) for j in range(3)]
    assert [r["id"] for r in rows] == [f"opener-{i}-{j}" for i in range(50000,50256) for j in range(3)]
    for a,b,c in zip(rows[::3],rows[1::3],rows[2::3]):
        assert [(x,y,z) for x,y,z in zip(a["prompt"],b["prompt"],c["prompt"],strict=True)
                if len({x,y,z})>1] == [("(","[","{")]
    labels = []
    for r in rows:
        x = r["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>")
        before,after = x.split("<|fim_suffix|>")
        parsed = []
        for i,symbol in enumerate((")","]","}",":")):
            try:
                ast.parse(before+symbol+after)
            except SyntaxError:
                continue
            parsed.append(i)
        assert parsed == [r["gold"]]
        labels.extend(parsed)
    labels = np.array(labels).reshape(256,3)
    solved, effects, brackets = {}, {}, {}
    with np.load(DATA / "crossed-scores.npz") as z:
        np.testing.assert_array_equal(labels.ravel(),z["gold"])
        for key in ("native","archived","ordinary","weighted"):
            x = z[key]
            assert x.shape == (256,3,3,4) and np.isfinite(x).all()
            solved[key] = (np.array([x[:, i, i].argmax(1) for i in range(3)]).T==labels).all(1)
            values = np.array([np.einsum("ijc,c->ij",x[:,:,a]-x[:,:,b],np.eye(4)[a]-np.eye(4)[b]).mean(1)
                               for a,b in combinations(range(3),2)]).T
            effects[key] = values.mean(1)
            brackets[key] = values.mean(0).tolist()
            np.testing.assert_allclose(brackets[key],result["pairs"][key],atol=1e-6)
        assert z["off"].shape == (768,4) and np.isfinite(z["off"]).all()
        solved["off"] = (z["off"].argmax(1).reshape(256,3)==labels).all(1)
    np.testing.assert_equal({k:int(v.sum()) for k,v in solved.items()}, result["complete"])
    picks = np.random.default_rng(445638).choice(256,(5000,256))
    def paired(v):
        return {"mean":float(v.mean()),"interval":np.quantile(v[picks].mean(1),[.025,.975]).tolist()}
    def validate(v,expected):
        d = paired(v)
        np.testing.assert_allclose([d["mean"],*d["interval"]],[expected["mean"],*expected["interval"]],atol=1e-6)
        return d
    metrics = {k:validate(v,result["effects"][k]) for k,v in effects.items()}
    halves = {k:validate(effects[k]-.5*effects["native"],v) for k,v in result["map_minus_half_native"].items()}
    weighted = validate(effects["weighted"]-effects["ordinary"],result["weighted_minus_ordinary"])
    ordinary_q = metrics["native"]["interval"][0]>0 and halves["ordinary"]["interval"][0]>0
    weighted_q = metrics["native"]["interval"][0]>0 and halves["weighted"]["interval"][0]>0
    selected = "weighted" if weighted_q and (not ordinary_q or weighted["interval"][0]>0) else "ordinary" if ordinary_q else "stop"
    assert selected==result["decision"] and bool(weighted_q and weighted["interval"][0]>0)==result["weighting_helps"]
    print(json.dumps(dict(effects=metrics,pair_means=brackets,map_minus_half_native=halves,
                          weighted_minus_ordinary=weighted,complete=result["complete"],decision=selected,
                          ordinary_minus_native=paired(effects["ordinary"]-effects["native"]),
                          ordinary_minus_archived_complete=paired(solved["ordinary"].astype(int)-solved["archived"])),indent=2))


if __name__ == "__main__":
    main()
