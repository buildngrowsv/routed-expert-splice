"""Audit the native screen and independently recreate all intervention targets."""
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/"results"
sys.path.insert(0,str(HERE.parent/"numeric_composition_study"))
from verify import audit


def main():
    rows=json.loads((ROOT/"qualification.json").read_text())
    assert [r["id"] for r in rows]==list(range(92000,92128))
    for row in rows:
        assert row["shift"]==0
        for i,p in enumerate(row["prompts"]):
            for j,a in enumerate(row["answers"]):
                scope={}
                try:
                    exec(compile(p+a+"\n","<archived native qualification>","exec"),scope)
                except AssertionError:
                    assert i!=j
                else:
                    assert i==j and scope["fetch"]()==int(a)
    with np.load(ROOT/"qualification.npz") as z:
        assert set(z.files)=={"native"}
        scores=z["native"]
    assert scores.shape==(128,4,4) and np.isfinite(scores).all()
    idx=np.arange(4)
    truth=scores[:,idx,idx]
    alternative=np.max(np.where(np.eye(4,dtype=bool),-np.inf,scores),axis=2)
    n=int((truth>alternative).all(1).sum())
    report=audit(ROOT,"39538af4cfef39c536bb695d56c8a3ec216927b6",512,93000,(94000,95000))
    assert report["screen_eligible"]==n and report["screen_groups"]==128 and n>=20
    programs=json.loads((ROOT/"programs.json").read_text())
    assert all(r["shift"]==0 and " * " not in r["prompts"][0] for r in programs["host"])
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
