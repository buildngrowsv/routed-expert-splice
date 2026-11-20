"""Measure a frozen donor advantage in marked and ordinary next-token formats."""
import json
import os
from pathlib import Path

import numpy as np

from bridge_input import interval
from large_opener import MODELS, model_files, score
from opener_programs import programs


def variants(panel):
    result = {k:[] for k in ("fim","empty","prefix")}
    for row in panel:
        p,suffix = row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
        assert p and suffix
        result["fim"].append(row)
        result["empty"].append({**row,"prompt":"<|fim_prefix|>"+p+"<|fim_suffix|><|fim_middle|>"})
        result["prefix"].append({**row,"prompt":p})
    return result


def main():
    out = Path(os.environ["STUDIO_JOB_DIR"])
    rows = programs(256,start=80000)
    formats = variants(rows)
    gold = np.array([r["gold"] for r in rows])
    idx = np.random.default_rng(245822).integers(256,size=(5000,256))
    results, complete, counts = {}, {}, {}
    for key in ("host","donor"):
        path = model_files(key)
        for fmt,sample in formats.items():
            name = key+"_"+fmt
            v = score(path,sample)
            assert v.shape==(len(rows),4) and np.isfinite(v).all()
            results[name] = v
            predictions = v.argmax(1)
            complete[name] = (predictions==gold).reshape(-1,3).all(1)
            counts[name] = np.bincount(predictions,minlength=4).tolist()
    deltas = {fmt:complete["donor_"+fmt].astype(int)-complete["host_"+fmt] for fmt in formats}
    answers = {fmt:interval(v,idx) for fmt,v in deltas.items()}
    prefix = answers["prefix"]
    qualifies = prefix["mean"]>=.10 and prefix["interval"][0]>0
    np.savez_compressed(out/"format-scores.npz",**results,gold=gold)
    (out/"programs.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    report = dict(decision="portable_advantage" if qualifies else "format_specific_or_unresolved",models=MODELS,
                  complete={k:int(v.sum()) for k,v in complete.items()},candidate_counts=counts,
                  gaps=answers,prefix_minus_fim_gap=interval(deltas["prefix"]-deltas["fim"],idx))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
