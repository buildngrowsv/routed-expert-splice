"""Independently execute every held-out patch and check solve and routing counts on CPU."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent/"results"


def main():
    summary=json.loads((R/"test.json").read_text())
    gates=json.loads((R/"gates.json").read_text())
    secondary=json.loads((R/"partial-test.json").read_text())
    for name,problemname,logname in (("clean","test-problems","test-logits"),("partial","partial-problems","partial-logits")):
        rows=json.loads((R/(problemname+".json")).read_text())
        with np.load(R/(logname+".npz")) as z:
            assert len(rows)==512 and len(set(r["prompt"] for r in rows))==512
            np.testing.assert_array_equal([r["gold"] for r in rows],z["gold"])
            for row in rows:
                prefix,post=row["prompt"].split("<|fim_suffix|>")
                prefix=prefix.split("<|fim_prefix|>")[1]
                tail=post.split("\n# Contract")[0]
                observed=[]
                for n in "abcd":
                    env={}
                    exec(compile(prefix+n+tail,"<held-out program>","exec"),env)
                    observed.append([env["select"](x) for x in row["hidden"]]==row["expected"])
                assert observed==row["passes"] and sum(observed)==1
            h,c=z["host"],z["donor"]
            gold=z["gold"]
            if name=="clean":
                sort=np.sort(h,axis=1)
                called=sort[:,-1]-sort[:,-2]<=gates["router_threshold"]
                expected=np.where(called[:,None],c,h)
                np.testing.assert_array_equal(z["routed"],expected)
                for key in ("host","donor","routed","native-101"):
                    solved=z[key].argmax(1)==gold
                    assert summary[key]==dict(solves=int(solved.sum()),pairs=int(solved.reshape(-1,2).all(1).sum()))
            else:
                mask=z["public_pass"]
                assert (mask.sum(1)==2).all()
                called=~mask[np.arange(512),c.argmax(1)]
                np.testing.assert_array_equal(called,z["called"])
                preds=dict(donor=c.argmax(1),host=h.argmax(1),donor_checked=np.where(mask,c,-np.inf).argmax(1),
                           host_checked=np.where(mask,h,-np.inf).argmax(1),first_consistent=mask.argmax(1))
                preds["cascade"]=np.where(called,preds["host_checked"],preds["donor"])
                assert {k:int((v==gold).sum()) for k,v in preds.items()}==secondary["solves"]
    print("Verified four executable patches on all 1,024 problems and both routing decisions.")


if __name__=="__main__": main()
