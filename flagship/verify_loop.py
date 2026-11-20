"""Recompute the held-out loop result using the saved traces and pinned tokenizer."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer


ARMS = ("native", "bridge", "wrong", "orthogonal", "dense", "whole")
PROGRAM = re.compile(
    r"def fetch\(\):\n    reading_(\d+) = ([1-8])\n    result = (\d+)\n"
    r"    for _ in range\(reading_\1\):\n        result \+= ([1-4])\n"
    r"    return result, (\d+)\n\nassert fetch\(\) == \("
)


def verified_tokens(row, tok):
    a, b = [tuple(map(int, PROGRAM.fullmatch(p).groups())) for p in row["prompts"]]
    assert a[0] == b[0] == row["id"] and a[2:] == b[2:]
    assert [a[1], b[1]] == row["values"][:2] and len(set(row["values"])) == 3
    start, increment, audit = a[2:]
    answers = [str((start + sum([increment] * value), audit)) for value in row["values"]]
    assert row["answers"] == answers
    for prompt, stored in zip(row["prompts"], row["ids"], strict=True):
        prefix = tok.encode(prompt, add_special_tokens=False).ids
        assert prefix == stored
        for answer, suffix in zip(answers[:2], row["tokens"], strict=True):
            assert tok.encode(prompt + answer[1:-1], add_special_tokens=False).ids == prefix + suffix
    differing = [i for i, (x, y) in enumerate(zip(*row["ids"], strict=True)) if x != y]
    assert differing == [row["site"]]
    return row["tokens"]


def exact(trace, expected, tok):
    tokens, unique = zip(*trace, strict=True)
    assert len(tokens) == len(expected) + 1
    end = tok.decode([tokens[-1]], skip_special_tokens=False)
    trailing = end[1:].lstrip(" \t")
    return (list(tokens[:-1]) == expected and all(unique) and end.startswith(")")
            and (not trailing or trailing[0] in "\r\n#"))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "held-out-loop"
    files = ("receipt.json", "result.json", "programs.json", "loop-scores.json")
    records = {name: (directory / name).read_bytes() for name in files}
    receipt, result, panel, ledger = [json.loads(records[name]) for name in files]
    tok = Tokenizer.from_file(str(args.tokenizer))
    assert receipt["revision"] == "56c672d83651745c0c9eef764ac28efb31a4e57d"
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert [row["id"] for row in panel["loop"]] == list(range(220000, 220768))
    targets = [verified_tokens(row, tok) for row in panel["loop"]]
    won = {}
    for arm in ARMS:
        traces = ledger["traces"][arm]
        assert len(traces) == 2 * len(targets)
        expected = [tokens[i] for tokens in targets for i in ((0, 1) if arm == "native" else (1, 0))]
        success = [exact(trace, gold, tok) for trace, gold in zip(traces, expected, strict=True)]
        won[arm] = np.asarray(success).reshape(-1, 2).all(axis=1)
    for arm in ("orthogonal", "dense"):
        np.testing.assert_allclose(ledger["norms"]["bridge"], ledger["norms"][arm], rtol=1e-5, atol=1e-6)
    qualified = won["native"]
    n = int(qualified.sum())
    complete = {arm: int(won[arm].sum()) for arm in ARMS}
    matched = {arm: int(won[arm][qualified].sum()) for arm in ARMS}
    drawn = np.random.default_rng(833721).integers(n, size=(5000, n))
    contrasts = {}
    for arm in ("wrong", "orthogonal", "dense"):
        diff = won["bridge"][qualified].astype(int) - won[arm][qualified]
        contrasts[arm] = dict(mean=float(diff.mean()), interval=np.percentile(diff[drawn].mean(axis=1), [2.5, 97.5]).tolist())
    saved = result["comparisons"]["loop"]
    assert (saved["groups"], saved["qualified"]) == (len(targets), n)
    assert complete == saved["total_success"] and matched == saved["qualified_success"]
    for arm, effect in contrasts.items():
        np.testing.assert_allclose(effect["mean"], saved["bridge_minus"][arm]["mean"])
        np.testing.assert_allclose(effect["interval"], saved["bridge_minus"][arm]["interval"])
    passes = (n >= 384 and matched["whole"] / n >= .7 and matched["bridge"] / n >= .7
              and all(contrasts[a]["mean"] >= .2 and contrasts[a]["interval"][0] > 0 for a in ("wrong", "orthogonal")))
    assert passes and result["decision"] == "exact_transfer_composes"
    dense = contrasts["dense"]
    assert result["beats_dense"] == (dense["mean"] >= .05 and dense["interval"][0] > 0)
    print(json.dumps(dict(groups=len(targets), qualified=n, total_success=complete, qualified_success=matched,
                          bridge_minus=contrasts, artifact_sha256={k: sha256(v).hexdigest() for k, v in records.items()},
                          tokenizer_sha256=sha256(args.tokenizer.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
