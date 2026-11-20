"""Independently audit the seven-block hold from the held and earlier token traces."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
OLD = HERE.parent / "numeric_carry_32b_study" / "results"
ORIGINAL = HERE.parent / "numeric_carry_study"
UTIL = runpy.run_path(str(ORIGINAL / "verify.py"))
MODES = ("native", "both", "random")


def first_error(rows, traces, terminators):
    endings = set(terminators.tolist())
    result = {}
    for row, scores in zip(rows, traces, strict=True):
        for original, trace in enumerate(scores):
            tokens = row["answer_tokens"]
            target = tokens[3 - original]
            n = len(target)
            before = len(tokens[original])
            if before == n:
                continue
            counts = result.setdefault(f"{before}_to_{n}", dict(first=0, second=0, terminator=0, correct=0))
            wrong = [i for i, v in enumerate(target) if trace[i, 0] != v or trace[i, 1] != 1]
            if wrong:
                counts[("first", "second")[wrong[0]]] += 1
            else:
                counts["correct" if trace[n, 0] in endings else "terminator"] += 1
    return result


def main():
    receipt = json.loads((ROOT / "receipt.json").read_text())
    assert receipt["revision"] == "708db914683b86fcb7b1acf02fca18fdfd87131f"
    assert receipt["job"] == "qwen-fixed-two-operands-32b" and receipt["status"] == "done" and receipt["exit"] == 0
    record = json.loads((ROOT / "result.json").read_text())
    baseline = json.loads((OLD / "result.json").read_text())
    assert record["hold_blocks"] == 7 and record["gain"] == baseline["gain"]
    payload = (ORIGINAL / "results/programs.json").read_bytes()
    assert sha256(payload).hexdigest() == record["prompts_digest"] == baseline["prompts_digest"]
    map_path = HERE.parent / "numeric_absolute_study/results/map.npz"
    assert sha256(map_path.read_bytes()).hexdigest() == record["map_digest"] == baseline["map_digest"]
    rows = json.loads(payload)["host"]
    UTIL["verify_programs"](rows)
    decode = UTIL["decode_hits"]
    with np.load(ROOT / "held-32b.npz") as values, np.load(OLD / "carry-32b.npz") as prior:
        assert set(values.files) == {*MODES, "terminators"}
        new = decode(rows, {m: values[m] for m in MODES}, values["terminators"])
        old = decode(rows, {m: prior[m] for m in (*MODES, "whole")}, prior["terminators"])
        first_errors = {"hold": first_error(rows, values["both"], values["terminators"]),
                        "one_shot": first_error(rows, prior["both"], prior["terminators"])}
    original_native = old["native"].all(1)
    frozen_native = new["native"].all(1)
    eligible = original_native & frozen_native
    n = int(eligible.sum())
    assert [int(mask.sum()) for mask in (original_native, frozen_native, eligible)] == [
        record[k] for k in ("native", "native_held", "eligible")]
    arms = {"hold": new["both"], "sham": new["random"], "one_shot": old["both"], "whole": old["whole"]}
    passed = {name: matrix[eligible].all(1) for name, matrix in arms.items()}
    assert record["counts"] == {key: int(value.sum()) for key, value in passed.items()}
    lengths = np.array([[len(v) for v in row["answers"]] for row in rows])
    for name, a, b in (("one_to_two", 1, 2), ("two_to_one", 2, 1)):
        mask = eligible[:, None] & (lengths == a) & (lengths[:, ::-1] == b)
        assert record["directions"][name] == dict(total=int(mask.sum()),
            correct={k: int(arm[mask].sum()) for k, arm in arms.items()})
    draws = np.random.default_rng(781394).integers(n, size=(5000, n))
    effects = {}
    for name in ("one_shot", "sham"):
        a = passed["hold"].astype(int) - passed[name]
        ci = np.percentile(a[draws].mean(1), [2.5, 97.5])
        effects[name] = (float(a.mean()), float(ci[0]))
        np.testing.assert_allclose([a.mean(), *ci], [record["contrasts"][name]["mean"],
                                                       *record["contrasts"][name]["interval"]])
    improved = (n >= .95 * record["native"] and effects["one_shot"][0] >= .05 and effects["one_shot"][1] > 0
                and effects["sham"][0] >= .2 and effects["sham"][1] > 0)
    assert record["decision"] == ("held_reference_improves" if improved else "unresolved")
    assert record["hit_original_target"] == (record["counts"]["hold"] >= .8*n)
    keep, single = passed["hold"], passed["one_shot"]
    overlap = {"both": int((keep & single).sum()), "hold_only": int((keep & ~single).sum()),
               "one_shot_only": int((~keep & single).sum()), "neither": int((~keep & ~single).sum())}
    print(json.dumps(dict(**record, overlap=overlap, first_errors=first_errors), indent=2))


if __name__ == "__main__":
    main()
