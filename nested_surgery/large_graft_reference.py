"""Measure intact donor headroom on the very same fresh repairs as the graft."""
from pathlib import Path
import json
import os
import numpy as np
from large_opener import model_files, score
from opener_programs import programs


def main():
    out=Path(os.environ["STUDIO_JOB_DIR"])
    graft=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs/large-qwen-block27-graft"
    rows=json.loads((graft/"programs.json").read_text())
    assert rows==programs(256,start=30000)
    gold=np.array([r["gold"] for r in rows])
    logits={k:score(model_files(k),rows) for k in ("host","donor")}
    np.testing.assert_array_equal(logits["host"].argmax(1),np.load(graft/"before-native.npy").argmax(1))
    np.savez_compressed(out/"standalone.npz",**logits,gold=gold)
    clean={k:(v.argmax(1)==gold).reshape(-1,3).all(1) for k,v in logits.items()}
    published=json.loads((graft/"result.json").read_text())
    strict=set(published["strict_ids"])
    opportunities=np.flatnonzero(clean["donor"] & ~clean["host"])
    held={rows[3*i]["id"].rsplit("-",1)[0] for i in opportunities}
    report=dict(decision="complete",groups=len(rows)//3,full_donor=int(clean["donor"].sum()),native=int(clean["host"].sum()),
                standalone_donor_only=sorted(held),strict_graft_new_on_donor_only=sorted(strict & held),
                strict_graft_new_elsewhere=sorted(strict-held))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
