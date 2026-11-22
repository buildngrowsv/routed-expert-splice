"""prog-e5-grow-frozen / prog-e5-grow-unfrozen: build the network from the bottom up.

Stage 1 trains a 4-block model. Stage 2 adds four blocks on top whose output projections start
at zero, so the model computes the same function when they arrive. Stage 3 adds four more.
In the frozen arm the lower blocks and the embedding stop training once covered, so they
become a read-only foundation (a tunnel the later blocks can only read); the final norm and
unembedding keep training. In the unfrozen arm every weight keeps training (progressive
stacking). Each stage gets one third of the dense E3 arm's tokens on the same mixture order.

    python run.py e5.grow --mode frozen
"""
from __future__ import annotations

import argparse
import json
import math
import shutil

import mlx.core as mx
import numpy as np

from common.job import Job, read_json, write_json
from common.paths import job_output
from e3 import configs, data, measure
from e3.model import DenseBlock, Model, count_params
from e3.trainer import evaluate, save_model, train

STAGES = (4, 8, 12)


def dense_reference(path) -> dict:
    """Final losses and the loss-versus-FLOPs curve of the E3 dense arm."""
    summary = read_json(path / "summary.json")
    rows = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("arm", "").startswith("dense") and r["step"] > 0]
    flops = np.array([r["flops"] for r in rows])
    mean = np.array([np.mean([r[f"val_{d}"] for d in data.TRAIN_DOMAINS]) for r in rows])
    return dict(summary=summary, flops=flops, mean=mean)


def add_blocks(model, cfg, n: int) -> None:
    scale = 1 / math.sqrt(2 * cfg.layers)
    for _ in range(n):
        b = DenseBlock(cfg, scale)
        b.wo = mx.zeros_like(b.wo)
        b.wd = mx.zeros_like(b.wd)
        model.blocks.append(b)
    mx.eval(model.parameters())


def pad(states: np.ndarray, total: int) -> np.ndarray:
    """Absent upper blocks pass the stream through unchanged (they start as identities)."""
    if states.shape[0] >= total:
        return states
    return np.concatenate([states, np.repeat(states[-1:], total - states.shape[0], axis=0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["frozen", "unfrozen"])
    ap.add_argument("--tokens", type=int, default=None, help="total; default equals the E3 dense arm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    name = f"prog-e5-grow-{a.mode}" + ("-smoke" if a.smoke else "")
    job = Job(name, vars(a))
    dense_dir = job.dir.parent / "prog-e3-dense-s1-smoke" if a.smoke else job_output("prog-e3-dense")
    ref = dense_reference(dense_dir) if (dense_dir / "summary.json").exists() else None
    total = a.tokens or (ref["summary"]["tokens"] if ref else 1_000_000_000)
    if a.smoke:
        total = 30 * configs.TOKENS_PER_STEP
    steps_total = total // configs.TOKENS_PER_STEP
    per_stage = steps_total // len(STAGES)
    cfg = configs.config("main", "dense")
    mx.random.seed(a.seed)
    model = Model(cfg, layers=STAGES[0])
    mx.eval(model.parameters())
    mix = data.Mixture(steps_total * configs.PER_DOMAIN * configs.SEQ + 1, configs.SEQ, configs.PER_DOMAIN, a.seed)
    val = data.validation(configs.SEQ, 8 if a.smoke else 64)
    probe_ids, probe_labels = data.probes(24)
    arm = f"grow_{a.mode}"
    frames, stage_rows, flops_done = {}, [], 0.0
    stage1_end = None
    for k, layers in enumerate(STAGES):
        if k > 0:
            add_blocks(model, cfg, layers - STAGES[k - 1])
            if a.mode == "frozen":
                for b in model.blocks[:STAGES[k - 1]]:
                    b.freeze()
                model.freeze(keys=["embed"], recurse=False)
        trainable_layers = layers - (STAGES[k - 1] if (k > 0 and a.mode == "frozen") else 0)
        fpt = configs.flops_per_token(cfg, layers, trainable_layers)
        offset = k * per_stage
        res = train(job, model, arm=f"{arm}-stage{k + 1}", steps=per_stage, peak_lr=configs.PEAK_LR["main"],
                    batches=lambda s, o=offset: mix.get(o + s), val=val, out=job.dir / f"stage{k + 1}",
                    tokens_per_step=configs.TOKENS_PER_STEP, eval_every=10 if a.smoke else max(50, per_stage // 8),
                    snapshots=3, probe_ids=probe_ids, flops_per_token=fpt, micro=configs.MICRO["main"])
        for s, st in res["snapshots"].items():
            if k > 0 and s == 0:
                continue
            frames[offset + s] = pad(st, STAGES[-1] + 1)
        if k == 0:
            stage1_end = frames[max(frames)]
        flops_done += per_stage * configs.TOKENS_PER_STEP * fpt
        final = res["history"][-1]
        stage_rows.append(dict(stage=k + 1, layers=layers, trainable_layers=trainable_layers, steps=per_stage,
                               flops=flops_done, val={d: final[f"val_{d}"] for d in data.TRAIN_DOMAINS + (data.GROWTH_DOMAIN,)},
                               mean_val=float(np.mean([final[f"val_{d}"] for d in data.TRAIN_DOMAINS])),
                               params=count_params(model)))
        job.log(stage_rows[-1])
    save_model(model, cfg, job.dir / "model", dict(mode=a.mode, stages=STAGES))

    final = stage_rows[-1]
    report = dict(mode=a.mode, stages=stage_rows, total_tokens=steps_total // len(STAGES) * len(STAGES) * configs.TOKENS_PER_STEP,
                  total_flops=flops_done)
    if ref:
        dense_final = ref["summary"]["train_mean_val"]
        flops, mean = ref["flops"], ref["mean"]
        at_equal_flops = float(np.interp(flops_done, flops, mean)) if len(flops) else float("nan")
        report.update(dense_final_mean_val=dense_final, dense_flops=float(flops[-1]) if len(flops) else None,
                      dense_mean_val_at_equal_flops=at_equal_flops,
                      gap_equal_tokens=final["mean_val"] / dense_final - 1,
                      gap_equal_flops=final["mean_val"] / at_equal_flops - 1)
        decision = "growth_matches" if report["gap_equal_flops"] <= 0.03 else "growth_costs"
    else:
        decision = "growth_costs"
        report["note"] = "dense reference missing"
    label = {"frozen": "Grown bottom-up, lower blocks frozen", "unfrozen": "Grown bottom-up, all blocks trainable"}[a.mode]
    measure.training_bundle(
        job.dir / "atlas", model=model, cfg=cfg, arm=arm, label=label,
        description=("Trained four blocks at a time from the bottom. " +
                     ("Each finished stage is frozen and later blocks only read it. " if a.mode == "frozen" else
                      "Every block keeps training as new ones are stacked on top. ") +
                     "Compare against the end of stage 1 to see which layers later stages changed."),
        probe_ids=probe_ids, probe_labels=probe_labels, snapshots=frames,
        compare=dict(label="Change since the end of stage 1", ref={arm: "stage1_end"}),
        extra_runs={"stage1_end": dict(states=stage1_end, condition="stage1_end", label="End of stage 1")},
        annotations=[dict(state=STAGES[0], kind="trained", label="Top of stage 1"),
                     dict(state=STAGES[1], kind="trained", label="Top of stage 2")])
    write_json(job.dir / "summary.json", report)
    for k in range(len(STAGES)):
        shutil.rmtree(job.dir / f"stage{k + 1}" / "ckpt", ignore_errors=True)  # resume state; model saved above
    job.finish(decision, **report)


if __name__ == "__main__":
    main()
