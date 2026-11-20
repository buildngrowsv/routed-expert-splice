"""New executable templates for the held-out nesting task."""
import ast
import random
from hashlib import sha256

PATCHES = (")", "]", "}", ":")
def panel(seed, count, depths):
    rows = []
    for i in range(count):
        rng = random.Random(f"nested-v1-{seed}-{i}")
        balanced = list(range(4))
        random.Random(f"nested-label-{seed}-{i//4}").shuffle(balanced)
        gold = balanced[i%4]
        expr = "x"
        shift = 0
        for _ in range(rng.randint(*depths)):
            step = rng.randint(-60, 60)
            shift += step
            expr = rng.choice(("({} + ({}))", "sum(({}, {}))", "[{}, 0][0] + ({})")).format(expr, step)
            expr = "(" + expr + ")"
        header = "def transform(x):\n    "
        prefix = ("return sum((1, "+expr+")", "return [0, "+expr,
                  "return {0: "+expr, "if "+expr+" < 0")[gold]
        tail = ("\n", "[1]\n", "[0]\n", "\n        return 1\n    return 0\n")[gold]
        cases = [rng.randint(-200, 200) for _ in range(10)] + [-shift-1,-shift+1]
        def oracle(x):
            return (x+shift+1,x+shift,x+shift,int(x+shift<0))[gold]
        expected = [oracle(x) for x in cases]
        head = header + prefix
        valid = []
        for patch in PATCHES:
            code = head + patch + tail
            try: parsed = ast.parse(code)
            except SyntaxError: continue
            scope = {"__builtins__":{"sum":sum}}
            exec(compile(parsed,"<nested-generated>","exec"),scope)
            assert [scope["transform"](x) for x in cases] == expected
            valid.append(patch)
        assert valid == [PATCHES[gold]]
        prompt = "<|fim_prefix|>"+head+"<|fim_suffix|>"+tail+"<|fim_middle|>"
        rows.append(dict(id=f"nested-{seed}-{i}",family="delimiters",gold=gold,prompt=prompt,
                         hidden=list(zip(cases,expected)),hash=sha256(prompt.encode()).hexdigest()))
    assert len({r["hash"] for r in rows}) == count
    return rows

