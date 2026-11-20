"""Apply a donor feature from different code at either of two host sources."""

import gc
from hashlib import sha256
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES, make_pairs
from identifier_binding import evaluate, programs, summarize
from identifier_transfer import gather, transform
from large_opener import MODELS, model_files

MAP_JOB = "qwen-transfer-identifier-subspace-dose-fix"
MAP_REVISION = "27a2e3bbccfd7eeda6a8acdc6bf46a5ae69ed934"


def load_axes(run):
    receipt = json.loads((run/"receipt.json").read_text())
    assert receipt["revision"]==MAP_REVISION and receipt["status"]=="done"
    assert json.loads((run/"result.json").read_text())["decision"]=="transfer"
    source = run/"map.npz"
    with np.load(source) as data:
        axes = [mx.array(data[k]) for k in ("donor_axis","host_axis","random_host_axis","mapping")]
    return axes,sha256(source.read_bytes()).hexdigest()


def panels(donor_tokenizer, host_tokenizer):
    host_rows = programs(host_tokenizer,256,40000)
    letters = [r["letters"] for r in host_rows[::2]]
    donor_rows = make_pairs([donor_tokenizer],len(letters),50000,letter_pairs=letters)
    assert [r["letters"] for r in donor_rows]==letters
    return donor_rows,host_rows


def main():
    mx.set_cache_limit(2*1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    axes,digest = load_axes(Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/MAP_JOB)
    paths = {m:model_files(m) for m in SITES}
    donor_rows,host_rows = panels(*(Tokenizer.from_file(str(paths[m]/"tokenizer.json")) for m in ("donor","host")))
    donor,_ = load_model(paths["donor"],strict=True)
    states = gather(donor,donor_rows,SITES["donor"]).astype(np.float32)
    translated,_ = transform(mx.array(states),*axes)
    translated = np.array(translated).reshape(len(donor_rows),2,-1)
    del donor
    gc.collect()
    host,_ = load_model(paths["host"],strict=True)
    scores,doses = evaluate(host,host_rows,None,SITES["host"],donor_changes=translated)
    report = summarize(scores)
    np.savez_compressed(out/"cross-binding.npz",perturbation_norms=doses, **scores)
    (out/"programs.json").write_text(json.dumps(dict(donor=donor_rows,host=host_rows),separators=(",",":"))+"\n")
    (out/"result.json").write_text(json.dumps(dict(models=MODELS,map_digest=digest,map_job=MAP_JOB,
                                                  groups=len(donor_rows),rank=25,**report),indent=2)+"\n")


if __name__ == "__main__":
    main()
