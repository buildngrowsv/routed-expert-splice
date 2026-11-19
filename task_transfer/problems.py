"""Deterministic executable patch problems. All executable fragments come from this file."""
import hashlib
import json
import operator
import random
import sqlite3

FAMILIES = ("bits", "lists", "sqlite")
CHOICES = {"bits": (" &", " |", " ^", " <<"), "lists": (" sum", " min", " max", " len"),
           "sqlite": (" SUM", " MIN", " MAX", " COUNT")}
SIZES = {"discovery":128, "confirmation":256, "train":1536, "val":256, "test":384}
BIT_FNS = (operator.and_, operator.or_, operator.xor, operator.lshift)
REDUCE = (sum, min, max, len)


def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def problem(family, split, i):
    key = f"{family}-{split}-{i:04}"
    rng = random.Random(seed_for(key))
    # Shuffle labels within each independent four-item stratum, without disclosing the ID to the model.
    labels = list(range(4))
    random.Random(seed_for(f"{family}-{split}-labels-{i//4}")).shuffle(labels)
    gold = labels[i%4]
    if family == "bits":
        x, y = rng.choice([("flags","mask"),("left","right"),("word","bits"),("field","shift")])
        bias = rng.randrange(0, 5)
        head = f"def update({x}, {y}):\n    {x} = {x} + {bias}\n    return {x}"
        tail = f" {y}"
        def sample():
            return [rng.randrange(2, 48), rng.randrange(1, 8)]
        def answer(case, op):
            return BIT_FNS[op](case[0] + bias, case[1])
        explain = f"Restore the operator in `return {x} ___ {y}`."
    else:
        threshold = rng.randrange(-6, 4)
        if family == "lists":
            name = rng.choice(("items","values","readings","samples"))
            delta = rng.randrange(-3,4)
            head = (f"def summarize({name}):\n    kept = [x + ({delta}) for x in {name} if x >= {threshold}]"
                    "\n    return")
            tail = "(kept)"
            explain = "Restore the built-in in `return ___(kept)`."
        else:
            table = rng.choice(("readings","events","sales","samples"))
            field = rng.choice(("amount","value","score","units"))
            head = "SELECT"
            tail = f"({field}) FROM {table} WHERE {field} >= {threshold}"
            explain = f"Repair the aggregate in SELECT ___{tail}. Table {table} has one integer column {field}."
            delta = 0
        def sample():
            vals = [rng.randrange(-9,17) for _ in range(rng.randrange(4,8))]
            vals.extend([rng.randrange(4,10),rng.randrange(11,19)])
            rng.shuffle(vals)
            return vals
        def answer(case, op):
            return REDUCE[op]([x+delta for x in case if x>=threshold])
    def cases(n):
        while True:
            xs = [sample() for _ in range(n)]
            ys = [answer(x,gold) for x in xs]
            if all(any(answer(x, op) != y for x,y in zip(xs,ys)) for op in range(4) if op != gold):
                return list(zip(xs,ys))
    shown, hidden = cases(3), cases(12)
    mark = "--" if family == "sqlite" else "#"
    lines = [f"{mark} {explain}", f"{mark} Examples (input -> result):"]
    lines.extend(f"{mark} {x} -> {y}" for x,y in shown)
    return dict(id=key,family=family,template=head+" ___"+tail,head=head,tail=tail,
                prompt="\n".join([*lines,head]),examples=shown,hidden=hidden,gold=gold)


def run_patches(p):
    # Execute four whitelisted patches and return a 12-case pass vector for each.
    checks = []
    for patch in CHOICES[p["family"]]:
        program = p["head"]+patch+p["tail"]
        if p["family"] == "sqlite":
            field = p["tail"].split(")")[0][1:]
            table = p["tail"].split(" FROM ")[1].split()[0]
            with sqlite3.connect(":memory:") as db:
                db.execute(f"CREATE TABLE {table} ({field} INTEGER)")
                verdict = []
                for xs,y in p["hidden"]:
                    db.execute(f"DELETE FROM {table}")
                    db.executemany(f"INSERT INTO {table} VALUES (?)", [(v,) for v in xs])
                    verdict.append(db.execute(program).fetchone()[0] == y)
        else:
            env = {"__builtins__":dict(zip(("sum","min","max","len"),REDUCE))}
            exec(compile(program,"<fixed patch>","exec"),env)
            name = "update" if p["family"] == "bits" else "summarize"
            verdict = [env[name](*xs if name=="update" else [xs]) == y for xs,y in p["hidden"]]
        checks.append(verdict)
    assert [all(v) for v in checks] == [j==p["gold"] for j in range(4)]
    return checks


def panel(family, split):
    rows = [problem(family, split, j) for j in range(SIZES[split])]
    assert len({r["prompt"] for r in rows}) == len(rows)
    return rows


def exact_p(wins, losses):
    from math import comb
    n = wins + losses
    return min(1., 2 * sum(comb(n,i) for i in range(min(wins,losses)+1)) / 2**n)
