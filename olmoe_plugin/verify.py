"""Recalculate the matched-budget and hot-swap results from the saved evidence."""

import json
from pathlib import Path


def main():
    evidence = json.loads(Path(__file__).with_name("evidence.json").read_text())
    revision = "4982b3395c2cace1c4295be8e0a47d93b099218a"
    for entry in evidence.values():
        receipt = entry["receipt"]
        assert receipt["revision"] == revision and receipt["status"] == "done" and receipt["exit"] == 0

    summary = evidence["summary"]["result"]
    base = evidence["census"]["result"]["base"]["val"]
    assert base == summary["base"]["val"] == summary["hot_swap"]["base"]["val"]
    names = {"plug": "plug", "fixed_router": "plug_fixed_router", "all_router": "plug_router_all",
             "lora": "lora", "full": "full", "welsh": "plug_welsh", "copied_down": "plug_copy"}
    for name, arm in names.items():
        entry = evidence[name]
        result = entry["result"]
        final = entry["evals"][str(result["steps"])]
        report = summary["table"][arm]
        assert result["base"]["val"] == base and result["final"]["val"] == final
        assert report["tokens"] == result["steps"] * 16 * 1024
        assert report["trainable"] == result["trainable_parameters"]
        domain = report["domain"]
        assert report["gain"] == base[domain] - final[domain]
        for held in ("english", "python"):
            assert report[f"{held}_change"] == final[held] / base[held] - 1
        if name != "full":
            unplug = result["unplug"]
            assert report["unplug_identical"] and unplug["identical"]
            assert unplug["max_abs_logit_diff"] == 0 and unplug["val"] == base

    plug, full = (evidence[arm]["evals"] for arm in ("plug", "full"))
    step = max(map(int, plug.keys() & full.keys()))
    gains = [base["icelandic"] - e[str(step)]["icelandic"] for e in (plug, full)]
    recovery = gains[0] / gains[1]
    assert evidence["full"]["result"]["stopped_early"]
    assert summary["matched"] == dict(step=step, plug_gain=gains[0], full_gain=gains[1], recovery=recovery)
    assert recovery >= .7 and summary["table"]["plug"]["python_change"] > .01
    assert summary["decision"] == "plugin_partial"

    swap = {name: r["val"] for name, r in summary["hot_swap"].items()}
    for language, arm in (("icelandic", "plug"), ("welsh", "welsh")):
        assert swap[f"plug_{language}"] == evidence[arm]["result"]["final"]["val"]
        assert swap[f"plug_{language}"][language] < swap["both"][language] < base[language]
    print(f"Verified {step * 16 * 1024:,} matched tokens, {recovery:.2%} recovery, and the plug-in partial decision.")
    print("Hot-swap losses and saved removal checks verified.")


if __name__ == "__main__":
    main()
