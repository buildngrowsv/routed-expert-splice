"""Balanced, behavior-changing indentation with disjoint structural splits."""
from itertools import product
import random

PATCHES=tuple(" "*n for n in (4,8,12,16))
families=list(product(range(4),repeat=3))
random.Random(78042).shuffle(families)
SPLITS={"train":families[:32],"discovery":families[32:40],"val":families[40:52],"test":families[52:]}
SIZE={"train":1536,"discovery":256,"val":384,"test":384}
DOMAIN=tuple(product(range(4),repeat=3))

def multiplier(form,value):
    return (int(value>0),value%2,value,max(0,3-value))[form]

def outcomes(forms,slots,base,bonus,x):
    values=[base+bonus];count=1
    for form,slot in zip(forms,slots):
        count*=multiplier(form,x[slot])
        values.append(base+bonus*count)
    return values

def panel(split):
    out=[]
    for layout in range(SIZE[split]//4):
        rng=random.Random(f"indent-v1-{split}-{layout}")
        forms=SPLITS[split][layout%len(SPLITS[split])]
        slots=rng.sample(range(3),3)
        base=rng.randint(-12,12);bonus=rng.choice([2,3,5,7,9])
        lines=["def tally(xs):",f"    total = {base}"]
        for i,(form,slot) in enumerate(zip(forms,slots)):
            x=f"xs[{slot}]"
            statement=(f"if {x} > 0:",f"if {x} % 2:",f"for n{i} in range({x}):",f"for n{i} in range(max(0, 3 - {x})):")[form]
            lines.append(" "*(4*(i+1))+statement)
        lines.append(" "*16+"pass")
        prefix="\n".join(lines)+"\n"
        suffix=f"total += {bonus}\n    return total\n"
        runs=[]
        for indent in PATCHES:
            env={}
            exec(compile(prefix+indent+suffix,"<controlled indentation>","exec"),env)
            runs.append(env["tally"])
        checks={x:outcomes(forms,slots,base,bonus,x) for x in DOMAIN}
        for x,answers in checks.items():
            assert [fn(x) for fn in runs]==answers
        for attempt in range(10000):
            cases=rng.sample(DOMAIN,16)
            if all(len({tuple(checks[x][d] for x in group) for d in range(4)})==4 for group in (cases[:4],cases[4:])):
                break
        else: raise RuntimeError("Could not construct a quartet")
        for d in range(4):
            tail=suffix+"\n"+"".join(f"assert tally({list(x)}) == {checks[x][d]}\n" for x in cases[:4])
            out.append(dict(id=f"{split}-{layout:04}-{d}",family="".join(map(str,forms)),gold=d,
                prompt="<|fim_prefix|>"+prefix+"<|fim_suffix|>"+tail+"<|fim_middle|>",
                hidden=[list(x) for x in cases[4:]],expected=[checks[x][d] for x in cases[4:]]))
    return out

