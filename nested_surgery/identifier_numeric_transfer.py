"""Translate an operand from a separate donor program into host arithmetic."""

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from identifier_baseline import SITES
from identifier_computation import DIGITS, programs
from identifier_transfer import evaluate, fit_map, gather
from large_opener import MODELS, model_files


def panels(tokenizers):
    cal = [programs(t,512,0,computed=False) for t in tokenizers]
    assert cal[0]==cal[1]
    host = programs(tokenizers[1],512,62000,computed=True)
    values = [r["source_values"] for r in host]
    donor = programs(tokenizers[0],512,70000,computed=False,values_by_pair=values)
    assert [r["source_values"] for r in donor]==values
    return cal[0],donor,host


def main():
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    paths={k:model_files(k) for k in SITES}
    tokens=[Tokenizer.from_file(str(paths[k]/"tokenizer.json")) for k in ("donor","host")]
    cal,src,dst=panels(tokens)
    donor,_=load_model(paths["donor"],strict=True)
    donor_fit=gather(donor,cal,SITES["donor"])
    donor_test=gather(donor,src,SITES["donor"])
    del donor
    gc.collect()
    host,_=load_model(paths["host"],strict=True)
    host_fit=gather(host,cal,SITES["host"])
    labels=[value for row in cal for value in row["labels"]]
    matrices=fit_map(donor_fit,host_fit,labels,classes=len(DIGITS))
    scores,dose,report=evaluate(host,dst,donor_test,[mx.array(a) for a in matrices])
    eligible=(scores["native"]>0).all(1)
    by_operator={}
    offsets=np.asarray([r["offset"] for r in dst])
    for name,mask in (("addition",offsets>0),("subtraction",offsets<0)):
        subset=eligible&mask
        by_operator[name]=dict(eligible=int(subset.sum()),
                               reversals={k:int((v[subset]<0).all(1).sum()) for k,v in scores.items() if k!="native"})
    np.savez_compressed(out/"numeric-transfer.npz",norms=dose,**scores)
    np.savez_compressed(out/"map.npz",**dict(zip(("donor_axis","host_axis","random_host_axis","mapping"),matrices)))
    (out/"programs.json").write_text(json.dumps(dict(donor=src,host=dst),separators=(",",":"))+"\n")
    report.update(models=MODELS,groups=len(dst),rank=len(DIGITS)-1,by_operator=by_operator)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
