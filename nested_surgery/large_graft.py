"""Put the frozen donor's final MLP in host block 27 and fit only block 28."""
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from large_opener import CACHE, PANEL, PATCHES, checkpoint_files, model_files
from opener_programs import programs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT/"task_transfer"))
from prepare import POS, ridge, write
from forced import Replacement
from penultimate.features import Zero, Affine
from delimiters import delimiter_panel, solved

GENERAL = ("Qwen/Qwen2.5-7B","d149729398750b98c0af14eb82c78cfe92750796")
ARMS = ("native","knockout","direct","shuffled","general","coder")
HOST_BLOCK = 26  # one-based block 27


def inputs(model, ids, block_idx):
    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for block in model.model.layers[:block_idx]:
        h = block(h, mask)
    block = model.model.layers[block_idx]
    residual = h + block.self_attn(block.input_layernorm(h), mask)
    return residual, block.post_attention_layernorm(residual)


def features(model, ids, at, native=False):
    xs, ys = [], []
    for start in range(0,len(ids),16):
        _, u = inputs(model, mx.array(ids[start:start+16,:64]), at)
        v = u[:,mx.array(POS)]
        xs.append(np.array(v.astype(mx.float32)).reshape(-1,v.shape[-1]))
        if native:
            z = model.model.layers[at].mlp(v)
            ys.append(np.array(z.astype(mx.float32)).reshape(-1,z.shape[-1]))
    return np.concatenate(xs), np.concatenate(ys) if native else None


def bridges(host, coder, general, out):
    weights_path = out/"interfaces.safetensors"
    modules = {"native":host.model.layers[HOST_BLOCK].mlp, "knockout":Zero()}
    if weights_path.exists():
        weights = mx.load(str(weights_path))
    else:
        data = np.load(CACHE.parent/"train-tokens.npz")["tokens"]
        manifest = json.loads((ROOT/"results/windows.json").read_text())
        digests = [row[-1] for row in manifest if row[0]=="train"]
        assert [sha256(a.tobytes()).hexdigest() for a in data]==digests
        # Different token segmentations would misalign the four registered positions.
        toks = [Tokenizer.from_file(str(p/"tokenizer.json")) for p in
                (CACHE.parent/"host",model_files("host"),model_files("donor"),checkpoint_files(*GENERAL))]
        for token in np.unique(data):
            labels = [t.id_to_token(int(token)) for t in toks]
            assert labels[0] is not None and len(set(labels))==1
        u, v = features(host, data, HOST_BLOCK, native=True)
        weights = {}
        for name, source in (("coder",coder),("general",general)):
            target,_ = features(source,data,len(source.model.layers)-1)
            w, bias = ridge(u,target)
            weights[name+"_w"],weights[name+"_b"] = mx.array(w), mx.array(bias)
        w,bias = ridge(u,v)
        weights["direct_w"],weights["direct_b"] = mx.array(w),mx.array(bias)
    modules["direct"] = Affine(weights["direct_w"],weights["direct_b"])
    for name, source, key in (("coder",coder,"coder"),("shuffled",coder,"coder"),("general",general,"general")):
        graft = Replacement(source.model.layers[-1].mlp,"shuffled" if name=="shuffled" else "intact",
                            weights[key+"_w"],weights[key+"_b"],None,None,Zero())
        if not weights_path.exists():
            parts = [np.array(graft.expert_output(mx.array(u[i:i+128])).astype(mx.float32)) for i in range(0,len(u),128)]
            w,bias = ridge(np.concatenate(parts), v)
            weights[name+"_out"],weights[name+"_bias"] = mx.array(w),mx.array(bias)
        graft.wout,graft.bout = weights[name+"_out"],weights[name+"_bias"]
        modules[name] = graft
    if not weights_path.exists():
        tmp = weights_path.with_name("interfaces.tmp.safetensors")
        mx.save_safetensors(str(tmp),weights)
        os.replace(tmp,weights_path)
    return modules


def encode(rows, tok):
    seq = [tok.encode(r["prompt"],add_special_tokens=False).ids for r in rows]
    choices = [tok.encode(p,add_special_tokens=False).ids for p in PATCHES]
    assert all(len(c)==1 for c in choices)
    lengths = np.array([len(s) for s in seq],dtype=np.int32)
    ids = np.zeros((len(rows),(max(lengths)+7)//8*8),dtype=np.int32)
    for i,s in enumerate(seq):
        ids[i,:len(s)] = s
        for patch, suffix in zip(PATCHES,choices):
            assert tok.decode(s+suffix,skip_special_tokens=False)==rows[i]["prompt"]+patch
    return ids,mx.array(lengths-1),mx.array([c[0] for c in choices]),mx.array([r["gold"] for r in rows])


@contextmanager
def replaced(block, graft):
    saved = block.mlp
    block.mlp = graft
    try:
        yield
    finally:
        block.mlp = saved


def evaluate(host, data, graft):
    ids, pos, choices,_ = data
    chunks = []
    with replaced(host.model.layers[HOST_BLOCK],graft):
        for i in range(0,len(ids),8):
            h = host.model(mx.array(ids[i:i+8]))
            last = h[mx.arange(len(h)),pos[i:i+8]]
            chunks.append(np.array(host.lm_head(last)[:,choices].astype(mx.float32)))
    return np.concatenate(chunks)


def finish(host, h, pos, choices):
    mask = create_attention_mask(h,None)
    for block in host.model.layers[HOST_BLOCK+1:]:
        h = block(h,mask)
    final = host.model.norm(h[mx.arange(len(pos)),pos])
    return host.lm_head(final)[:,choices].astype(mx.float32)


def train(host, mod, initial, training):
    ids, pos, choices, gold = training
    residual,u = inputs(host,mx.array(ids),HOST_BLOCK)
    entrance = mx.stop_gradient(residual + mod(u))
    mx.eval(entrance)
    received = host.model.layers[HOST_BLOCK+1]
    master = tree_map(lambda v:v.astype(mx.float32),initial)
    optimizer = optim.Adam(2e-5,bias_correction=True)
    def objective(w):
        received.update(tree_map(lambda v:v.astype(mx.bfloat16),w))
        return nn.losses.cross_entropy(finish(host,entrance,pos,choices),gold,reduction="mean")
    vg = mx.value_and_grad(objective)
    losses = []
    for _ in range(8):
        loss,grad = vg(master)
        master = optimizer.apply_gradients(grad,master)
        mx.eval(loss,master,optimizer.state)
        assert np.isfinite(loss.item())
        losses.append(loss.item())
    received.update(tree_map(lambda v:v.astype(mx.bfloat16),master))
    mx.eval(received.parameters())
    return losses


def run(out):
    import gc
    host_path = model_files("host")
    host,_ = load_model(host_path,strict=True)
    coder,_ = load_model(model_files("donor"),strict=True)
    general,_ = load_model(checkpoint_files(*GENERAL),strict=True)
    assert len(host.model.layers)==48 and len(coder.model.layers)==len(general.model.layers)==28
    rows = programs(256,start=30000)
    ids = json.loads((ROOT/"nested_surgery/results/fit/training_ids.json").read_text())
    pool = {r["id"]:r for r in delimiter_panel("train")}
    train_rows = [pool[i] for i in ids]
    for row in train_rows:
        solved(row)
    old = json.loads((PANEL/"programs.json").read_text())
    assert not ({r["prompt"] for r in rows} & {r["prompt"] for r in train_rows+old})
    assert len({r["prompt"] for r in rows})==768
    write(out/"programs.json",rows)
    write(out/"training_ids.json",ids)
    tok = Tokenizer.from_file(str(host_path/"tokenizer.json"))
    test = encode(rows,tok)
    training = encode(train_rows,tok)
    # Native scoring must reproduce the frozen qualification answers before using the splice.
    old_native = np.load(Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs/large-qwen-opener-qualification/host.npy")
    native = host.model.layers[HOST_BLOCK].mlp
    audit_data = encode(old[:8],tok)
    baseline = evaluate(host,audit_data,native)
    np.testing.assert_array_equal(baseline.argmax(1),old_native[:8].argmax(1))
    np.testing.assert_allclose(baseline,old_native[:8],atol=.25,rtol=.02)
    res,u = inputs(host,mx.array(audit_data[0]),HOST_BLOCK)
    native_cached = np.array(finish(host,res+native(u),audit_data[1],audit_data[2]))
    np.testing.assert_allclose(native_cached,baseline,atol=.25,rtol=.02)
    np.testing.assert_array_equal(native_cached.argmax(1),baseline.argmax(1))
    write(out/"scoring-audit.json",dict(items=8,full_logits_max_error=float(abs(baseline-native_cached).max()),
          qualification_logit_difference=float(abs(old_native[:8]-baseline).max()),block=27,train_block=28))
    arms=bridges(host,coder,general,out)
    # Only frozen donor MLPs inside the grafts are needed after fitting the bridges.
    del coder,general
    gc.collect()
    received = host.model.layers[HOST_BLOCK+1]
    initial = tree_map(lambda w:w,received.parameters())
    training_count = sum(x.size for _,x in tree_flatten(initial))
    def record(name,values):
        assert values.shape==(len(rows),4) and np.isfinite(values).all()
        tmp=out/(name+".partial.npy")
        np.save(tmp,values)
        os.replace(tmp,out/(name+".npy"))
    for name in ARMS:
        marker=out/(name+"-complete.json")
        if marker.is_file():
            continue
        received.update(initial)
        record("before-"+name,evaluate(host,test,arms[name]))
        losses = train(host,arms[name],initial,training)
        record(name,evaluate(host,test,arms[name]))
        if name=="coder":
            for replacement in ("knockout","native","shuffled"):
                record("coder-to-"+replacement,evaluate(host,test,arms[replacement]))
            ckpt=out/"block28-coder.partial.safetensors"
            received.save_weights(str(ckpt))
            os.replace(ckpt,out/"block28-coder.safetensors")
        write(marker,dict(arm=name,losses=losses,steps=8,trainable=training_count))
    from large_graft_report import summarize
    write(out/"result.json",summarize(out,rows))


if __name__=="__main__":
    mx.set_cache_limit(2*1024**3)
    if sys.argv[1:]==["prefetch-general"]:
        print(checkpoint_files(*GENERAL))
    else:
        run(Path(os.environ["STUDIO_JOB_DIR"]))
