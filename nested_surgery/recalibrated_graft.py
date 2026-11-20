"""Repeat the registered host adaptation with a qualified replacement input map."""
import gc
from hashlib import file_digest
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from bridge_input import interval
from large_graft import (ARMS, GENERAL, HOST_BLOCK, PANEL, ROOT, Replacement, Zero, bridges, checkpoint_files,
                         delimiter_panel, encode, evaluate, model_files, solved, train, write)
from large_opener import score
from opener_programs import programs


def group_report(scores, rows):
    shape = (len(rows)//3, 3)
    gold = np.array([row["gold"] for row in rows])
    individual = {k: v.argmax(1) == gold for k,v in scores.items()}
    good = {k: v.reshape(shape).all(1) for k,v in individual.items()}
    measured = good["refit"]
    indices = np.random.default_rng(688431).integers(len(measured), size=(5000,len(measured)))
    controls = ("native","refit-shuffled","coder")
    comparisons = {k:interval(measured.astype(int)-good[k],indices) for k in controls}
    new = measured & ~good["native"] & ~good["before-native"]
    strict = new & ~good["refit-to-knockout"] & ~good["refit-to-native"]
    donor_only = good["standalone-donor"] & ~good["before-native"]
    def names(mask):
        return [rows[j*3]["id"].rsplit("-",1)[0] for j in np.flatnonzero(mask)]
    advances = int(strict.sum())>=10 and all(v["mean"]>=.05 and v["interval"][0]>0 for v in comparisons.values())
    return dict(decision="replicate" if advances else "stop",groups=len(measured),
                individual={k:int(v.sum()) for k,v in individual.items()},complete={k:int(v.sum()) for k,v in good.items()},
                advantages=comparisons,strict_ids=names(strict),new_ids=names(new),
                strict_overlap_with_standalone_donor=names(strict & donor_only),
                strict_removed_by_old_input=names(strict & ~good["refit-to-old-input"]),
                off_rescues=names(good["refit-to-knockout"] & ~measured),
                off_losses=names(measured & ~good["refit-to-knockout"]),
                native_regressions=names(good["native"] & ~measured))


def run():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    runs = Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
    original = runs/"large-qwen-block27-graft"
    calibration = runs/"large-qwen-paired-input-calibration"
    assert json.loads((calibration/"result.json").read_text())["decision"]=="ordinary"
    original_maps = original/"interfaces.safetensors"
    refit_maps = calibration/"calibrated-maps.safetensors"
    hashes = {}
    for k,path in (("archived",original_maps),("refit",refit_maps)):
        with path.open("rb") as f:
            hashes[k] = file_digest(f,"sha256").hexdigest()
    assert hashes["archived"]==json.loads((calibration/"result.json").read_text())["interface_sha256"]
    new_map = mx.load(str(refit_maps))
    rows = programs(256,start=60000)
    training_ids = json.loads((ROOT/"nested_surgery/results/fit/training_ids.json").read_text())
    pool = {row["id"]:row for row in delimiter_panel("train")}
    training_rows = [pool[i] for i in training_ids]
    for r in training_rows:
        solved(r)
    fresh = {r["prompt"] for r in rows}
    for old_dir in (PANEL,original,runs/"large-qwen-native-vs-bridged-input",calibration):
        before = json.loads((old_dir/"programs.json").read_text())
        assert fresh.isdisjoint(r["prompt"] for r in before)
    assert fresh.isdisjoint(r["prompt"] for r in training_rows)
    assert fresh.isdisjoint(r["prompt"] for r in programs(512,start=45000))
    write(out/"programs.json",rows)
    write(out/"training_ids.json",training_ids)
    write(out/"inputs.json",dict(hashes=hashes,calibration="ordinary",host_block=27,trained_block=28))
    def save(k,values):
        assert values.shape==(len(rows),4) and np.isfinite(values).all()
        temporary = out/(k+".tmp.npy")
        np.save(temporary,values)
        temporary.replace(out/(k+".npy"))
    if not (out/"standalone-donor.npy").exists():
        save("standalone-donor",score(model_files("donor"),rows))
        gc.collect()
    host_p = model_files("host")
    host,_ = load_model(host_p,strict=True)
    coder,_ = load_model(model_files("donor"),strict=True)
    general,_ = load_model(checkpoint_files(*GENERAL),strict=True)
    assert len(host.model.layers)==48 and len(coder.model.layers)==len(general.model.layers)==28
    arms = bridges(host,coder,general,original)
    def calibrated(old, mode):
        return Replacement(old.expert,mode,new_map["ordinary_w"],new_map["ordinary_b"],old.wout,old.bout,Zero())
    arms["refit"] = calibrated(arms["coder"],"intact")
    arms["refit-shuffled"] = calibrated(arms["shuffled"],"shuffled")
    del coder, general, new_map
    gc.collect()
    tokenizer = Tokenizer.from_file(str(host_p/"tokenizer.json"))
    test = encode(rows,tokenizer)
    adaptation = encode(training_rows,tokenizer)
    audited_rows = json.loads((original/"programs.json").read_text())[:8]
    check = evaluate(host,encode(audited_rows,tokenizer),arms["native"])
    previous = np.load(original/"before-native.npy")[:8]
    np.testing.assert_array_equal(check.argmax(1),previous.argmax(1))
    np.testing.assert_allclose(check,previous,atol=.25,rtol=.02)
    receiver = host.model.layers[HOST_BLOCK+1]
    initial = tree_map(lambda w:w,receiver.parameters())
    trainable = sum(v.size for _,v in tree_flatten(initial))
    assert trainable==275268608
    removals = {"refit-to-knockout":"knockout","refit-to-native":"native","refit-to-old-input":"coder",
                "refit-to-shuffled":"refit-shuffled"}
    for name in (*ARMS,"refit-shuffled","refit"):
        marker = out/(name+"-complete.json")
        if marker.exists():
            continue
        receiver.update(initial)
        save("before-"+name,evaluate(host,test,arms[name]))
        losses = train(host,arms[name],initial,adaptation)
        save(name,evaluate(host,test,arms[name]))
        if name=="refit":
            for k,arm in removals.items():
                save(k,evaluate(host,test,arms[arm]))
            path = out/"block28-refit.tmp.safetensors"
            receiver.save_weights(str(path))
            path.replace(out/"block28-refit.safetensors")
        write(marker,dict(arm=name,losses=losses,trainable=trainable,steps=8))
    names = ["standalone-donor",*ARMS,"refit","refit-shuffled",*removals]
    names += ["before-"+k for k in (*ARMS,"refit","refit-shuffled")]
    scores = {k:np.load(out/(k+".npy")) for k in names}
    write(out/"result.json",group_report(scores,rows))


if __name__=="__main__":
    run()
