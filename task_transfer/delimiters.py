"""Executable one-token Python syntax repairs with independent behavioral oracles."""
import random

from problems import SIZES, seed_for

PATCHES = (")", "]", "}", ":")
BUILTINS = {f.__name__: f for f in (sum,min,max,len,sorted)}


def solved(row):
    checks = []
    for patch in PATCHES:
        env = {"__builtins__":BUILTINS}
        try:
            exec(compile(row["head"]+patch+row["tail"],"<syntax repair>","exec"),env)
            passed = [env["transform"](x)==y for x,y in row["hidden"]]
        except (SyntaxError,TypeError,ValueError,KeyError,IndexError):
            passed = [False]*len(row["hidden"])
        checks.append(passed)
    assert [all(v) for v in checks]==[k==row["gold"] for k in range(4)]
    return checks


def delimiter_panel(split, plain=False):
    rows = []
    for i in range(SIZES[split]):
        key = f"delimiters-{split}-{i:04}"
        rng = random.Random(seed_for(key))
        labels = list(range(4))
        random.Random(seed_for(f"delimiters-{split}-labels-{i//4}")).shuffle(labels)
        gold, style = labels[i%4], rng.randrange(3)
        a,b = rng.randrange(2,98),rng.randrange(2,98)
        name = rng.choice(("items","samples","counts","values","readings"))
        offset = rng.randrange(-65,67)
        header = f"def transform({name}):\n    data = [x + ({offset}) for x in {name}]\n    "
        if gold == 0:
            fn = (sum,max,min)[style]
            left, right = f"return {fn.__name__}(data", f" + {a} - {b}\n"
            target = lambda z: fn(z)+a-b
        elif gold == 1:
            idx = rng.randrange(3)
            left = ("return data[", "return sorted(data)[", "return data[::-1][")[style]+str(idx)
            right = f" * {a} - {b}\n"
            target = lambda z: (z,sorted(z),z[::-1])[style][idx]*a-b
        elif gold == 2:
            fn = (sum,max,min)[style]
            left = f"by_key = {{{a}: {fn.__name__}(data), -{b}: len(data)"
            right = f"\n    return by_key[{a}] + by_key[-{b}]\n"
            target = lambda z: fn(z)+len(z)
        else:
            op = ("<",">","==")[style]
            left = f"if sum(data) {op} {a}"
            right = f"\n        return {b}\n    return -{a}\n"
            target = lambda z: b if (sum(z)<a,sum(z)>a,sum(z)==a)[style] else -a
        def cases(n):
            result = []
            for j in range(n):
                xs = [rng.randint(-22,32) for _ in range(rng.randrange(3,8))]
                # Cover both branches, including equality, in the conditional templates.
                if gold==3 and j%3==0:
                    desired = a + ((-1,1,0)[style] if j%2==0 else (1,-1,1)[style])
                    xs[-1] = desired - len(xs)*offset - sum(xs[:-1])
                result.append((xs, target([x+offset for x in xs])))
            return result
        example, hidden = cases(1), cases(12)
        x,y = example[0]
        prefix = f"# Example: transform({x}) == {y}\n"+header+left
        if plain:
            prompt = "# Fill in the missing Python punctuation.\n"+prefix+"___"+right+"# Missing token:"
        else:
            prompt = "<|fim_prefix|>"+prefix+"<|fim_suffix|>"+right+"<|fim_middle|>"
        rows.append(dict(id=key,family="delimiters",gold=gold,style=style,head=prefix,tail=right,
                         prompt=prompt,examples=example,hidden=hidden))
    return rows
