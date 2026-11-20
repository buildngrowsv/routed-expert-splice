"""Recompute full-panel predictions and synchronized timing statistics."""
import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
root = here / "results"
frozen = here.parent / "frozen_graft_route_confirmation_study/results"
receipt = json.loads((root / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "large-qwen-frozen-live-route-timing", "4042f0fd1d2f1d84744a32571e19253912c7a3ed", "done", 0)
r = json.loads((root / "result.json").read_text())
bootstrap = np.random.default_rng(18851).integers(2304,size=(5000,2304))
samples = np.random.default_rng(15239).integers(32,size=(5000,32))
gold = np.arange(3)
audit = {}
with np.load(root / "live-route.npz") as evidence:
    for layout in ("neutral","decoys"):
        with np.load(frozen/f"{layout}.npz") as reference:
            pilot = reference["donor"].astype(float)
            graft = reference["graft"].astype(float)
            call = reference["calls"]
        actual = evidence[layout+"_actual"].reshape(-1,3)
        live_calls = evidence[layout+"_called"].reshape(-1,3)
        assert actual.shape == live_calls.shape == call.shape == (2304,3)
        assert np.isin(actual,[0,1,2,3]).all()
        estimated = np.where(call[...,None], (.75*pilot+.25*graft), pilot).argmax(-1)
        observed = r["layouts"][layout]
        assert (live_calls!=call).sum() == observed["call_changes"]
        assert (actual!=estimated).any(-1).sum() == observed["mismatched_groups"]
        assert live_calls.sum() == observed["actual_calls"] and live_calls.mean() == observed["call_fraction"]
        native = (pilot.argmax(-1)==gold).all(-1)
        live = (actual==gold).all(-1)
        assert native.sum() == observed["native"] and live.sum() == observed["complete"]
        delta = live.astype(int)-native
        np.testing.assert_allclose(delta.mean(),observed["lift"])
        np.testing.assert_allclose(np.quantile(delta[bootstrap].mean(1),[.025,.975]),observed["interval"])
        which=evidence[layout+"_timed_indices"]
        np.testing.assert_array_equal(which,np.sort(np.random.default_rng(88143).choice(576,32,replace=False)))
        outcomes=evidence[layout+"_timed_winners"]
        assert outcomes.shape==(32,3,12)
        np.testing.assert_array_equal(actual.ravel().reshape(576,12)[which],outcomes[:,2])
        times=evidence[layout+"_durations_ns"]
        assert times.shape==(32,3,3) and (times>0).all()
        np.testing.assert_allclose(np.median(times.reshape(-1,3),axis=0)/1e6,observed["median_ms"])
        med=np.median(times,axis=1)
        gains=med[:,1]/med[:,2]
        np.testing.assert_allclose(np.median(gains),observed["speedup_over_always"])
        np.testing.assert_allclose(np.quantile(np.median(gains[samples],axis=1),[.025,.975]),observed["speedup_interval"])
        audit[layout] = observed
d,n = audit["decoys"],audit["neutral"]
passed = (d["speedup_over_always"]>=1.5 and d["speedup_interval"][0]>1 and d["lift"]>=.03 and d["interval"][0]>0
          and n["interval"][0]>=-.02 and all(v["mismatched_groups"]<=.005*v["n"] and v["call_fraction"]<=.33
                                              for v in audit.values()))
assert r["decision"] == ("live_route_confirms_speed_and_quality" if passed else "unresolved")
print(json.dumps(audit,indent=2))
