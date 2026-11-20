"""Deterministic executable triplets for suffix-neutral missing-delimiter tests."""
import ast
import random
from panels import PATCHES

def programs(n, start=0):
    rows=[]
    for i in range(start,start+n):
        rng=random.Random(f"opener-transfer-v1-{i}")
        expr="x"
        for _ in range(rng.randint(4,6)):
            expr=f"({expr} + ({rng.randrange(-99,100)}))"
            expr=rng.choice(["("+expr+")","sum(("+expr+", 0))","["+expr+", 0][0]"])
        inputs=[rng.randrange(-100,100) for _ in range(12)]
        for k,c in enumerate("([{"):
            head="def transform(x):\n    value = "+c+expr
            tail="\n    return x\n"
            valid=[]
            for j,patch in enumerate(PATCHES):
                try:tree=ast.parse(head+patch+tail)
                except SyntaxError:continue
                env={"__builtins__":{"sum":sum}}
                exec(compile(tree,"<generated opener>","exec"),env)
                assert all(env["transform"](x)==x for x in inputs)
                valid.append(j)
            assert valid==[k]
            rows.append(dict(id=f"opener-{i}-{k}",family="delimiters",gold=k,
                prompt="<|fim_prefix|>"+head+"<|fim_suffix|>"+tail+"<|fim_middle|>"))
        assert sum(a!=b for a,b in zip(rows[-3]["prompt"],rows[-1]["prompt"]))==1
    return rows
