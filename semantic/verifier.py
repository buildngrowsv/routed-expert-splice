"""Evaluate a cheap donor-first partial-verification cascade on fresh examples."""
from pathlib import Path
import random
from experiment import R,NAMES,encode,full_logits,load_model,panel,paired_ci,write
import numpy as np


def main():
    rows=panel("ambiguous",256)
    fits=[]
    for i,r in enumerate(rows):
        # Only generated source through the suffix marker; replace the single public assertion.
        prefix,suffix=r["prompt"].split("<|fim_suffix|>")
        head=prefix.split("<|fim_prefix|>")[1]
        tail,rest=suffix.split("\n# Contract:")
        contract,assertion=rest.split("\nassert select(")
        field=int(contract.split("values[")[1].split("]")[0])
        bias=int(contract.split("+ (")[1].split(")")[0])
        rng=random.Random(6181+i//2)
        x=rng.sample(range(-999,1000),4)
        other=rng.choice([j for j in range(4) if j!=field])
        x[other]=x[field]
        value=x[field]+bias
        same=[]
        for n in NAMES:
            env={}
            exec(compile(head+n+tail,"<binding>","exec"),env)
            same.append(env["select"](x)==value)
        assert sum(same)==2 and same[r["gold"]]
        r["prompt"]=prefix+"<|fim_suffix|>"+tail+"\n# Contract:"+contract+f"\nassert select({x}) == {value}\n<|fim_middle|>"
        r["public_pass"]=same
        fits.append(same)
    data=encode(rows,NAMES)
    h,c=[full_logits(load_model(n),data) for n in ("host","donor")]
    passed=np.array(fits)
    normal=c.argmax(1)
    call=~passed[np.arange(len(rows)),normal]
    def checked(scores): return np.where(passed,scores,-np.inf).argmax(1)
    preds=dict(donor=normal,host=h.argmax(1),donor_checked=checked(c),host_checked=checked(h),
               first_consistent=passed.argmax(1),cascade=np.where(call,checked(h),normal))
    gold=data[-1]
    ok={k:p==gold for k,p in preds.items()}
    report=dict(n=len(gold),second_calls=int(call.sum()),solves={k:int(v.sum()) for k,v in ok.items()},
           cascade_minus_checked_donor=paired_ci(ok["cascade"],ok["donor_checked"]),
           gained=int((ok["cascade"]&~ok["donor_checked"]).sum()),lost=int((~ok["cascade"]&ok["donor_checked"]).sum()))
    write(R/"partial-test.json",report)
    write(R/"partial-problems.json",rows)
    np.savez_compressed(R/"partial-logits.npz",host=h,donor=c,gold=gold,public_pass=passed,called=call)
    print(report,flush=True)


if __name__=="__main__": main()
