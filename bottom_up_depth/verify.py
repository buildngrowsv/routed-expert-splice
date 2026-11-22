"""Independently check progressive-growth compute and the dense interpolation."""

from bisect import bisect_right
import json
from math import isclose
from pathlib import Path
from statistics import mean


REVISION = "4982b3395c2cace1c4295be8e0a47d93b099218a"
DOMAINS = ("english", "code", "welsh")
TOKENS_PER_STEP = 120 * 512
BLOCK = 4 * 768**2 + 3 * 768 * 2048 + 2 * 512 * 768
HEAD = 768 * 8192


def flop_rate(layers, trainable):
    return 2 * (layers * BLOCK + HEAD) + 4 * (trainable * BLOCK + HEAD)


def main():
    data = json.loads(Path(__file__).with_name("evidence.json").read_text())
    for entry in data.values():
        receipt = entry["receipt"]
        assert receipt["revision"] == REVISION and receipt["status"] == "done" and receipt["exit"] == 0
    dense = data["dense"]["result"]
    assert dense["flops_per_token"] == flop_rate(12, 12)
    curve = data["dense"]["curve"]
    values = [mean(row[f"val_{d}"] for d in DOMAINS) for row in curve]
    flops = [row["step"] * TOKENS_PER_STEP * flop_rate(12, 12) for row in curve]
    assert flops == [row["flops"] for row in curve]
    assert curve[-1]["step"] == dense["steps"] == 6510
    assert isclose(values[-1], dense["train_mean_val"], abs_tol=1e-12)

    for name, trainable in (("frozen", (4, 4, 4)), ("unfrozen", (4, 8, 12))):
        r = data[name]["result"]
        cost = 0
        for n, (stage, layers, train) in enumerate(zip(r["stages"], (4, 8, 12), trainable, strict=True), 1):
            assert (stage["stage"], stage["layers"], stage["trainable_layers"], stage["steps"]) == (n, layers, train, 2170)
            cost += stage["steps"] * TOKENS_PER_STEP * flop_rate(layers, train)
            assert cost == stage["flops"]
            assert isclose(mean(stage["val"][d] for d in DOMAINS), stage["mean_val"], abs_tol=1e-12)
        assert cost == r["total_flops"] and r["total_tokens"] == dense["tokens"] == 6510 * TOKENS_PER_STEP
        after = bisect_right(flops, cost)
        before = after - 1
        ratio = (cost - flops[before]) / (flops[after] - flops[before])
        matched = values[before] + ratio * (values[after] - values[before])
        end = r["stages"][-1]["mean_val"]
        equal_compute = end / matched - 1
        equal_tokens = end / values[-1] - 1
        assert isclose(matched, r["dense_mean_val_at_equal_flops"], abs_tol=1e-12)
        assert isclose(equal_compute, r["gap_equal_flops"], abs_tol=1e-12)
        assert isclose(equal_tokens, r["gap_equal_tokens"], abs_tol=1e-12)
        assert r["decision"] == ("growth_matches" if equal_compute <= .03 else "growth_costs")
        print(f"{name}: {cost / flops[-1]:.2%} of full compute; loss gap {equal_compute:+.2%} at matched FLOPs, "
              f"{equal_tokens:+.2%} at matched tokens")


if __name__ == "__main__":
    main()
