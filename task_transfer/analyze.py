"""CPU-only reconstruction of solved programs, paired effects and the removal ledger."""
from pathlib import Path
import hashlib
import json

import numpy as np

from delimiters import PATCHES, delimiter_panel, solved
from problems import exact_p, SIZES

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)
ARMS=("native","knockout","direct","shuffled","general","coder")


def write(name, obj):
    (R/name).write_text(json.dumps(obj,indent=2)+"\n")


def paired(diff):
    """Two-stage paired bootstrap of order seeds and problem IDs within this generator."""
    diff=np.atleast_2d(diff).astype(float)
    rng=np.random.default_rng(72558)
    estimates=[]
    for _ in range(20):
        item=rng.integers(diff.shape[1],size=(250,diff.shape[1]))
        seed=rng.integers(diff.shape[0],size=(250,diff.shape[0]))
        x=diff[seed[:,:,None],item[:,None,:]]
        estimates.extend(x.mean((1,2)).tolist())
    return dict(mean=float(diff.mean()),interval=np.quantile(estimates,[.025,.975]).tolist())


def main():
    rows=json.loads((R/"test-problems.json").read_text())
    keys=np.array([r["id"] for r in rows])
    n=len(rows)
    seen=[set(),set()]
    for split in SIZES:
        group=delimiter_panel(split)
        for r in group:
            src=r["head"].split("\n",1)[1]+PATCHES[r["gold"]]+r["tail"]
            for registry,text in zip(seen,(r["prompt"],src)):
                dig=hashlib.sha256(text.encode()).digest()
                assert dig not in registry,(r["id"],"duplicate")
                registry.add(dig)
    assert [r["prompt"] for r in rows]==[r["prompt"] for r in delimiter_panel("test")]
    gold=np.array([r["gold"] for r in rows])
    oracle=np.array([solved(r) for r in rows]).all(-1)
    for j,row in enumerate(rows):
        np.testing.assert_array_equal(np.array(row["candidate_tests"]).all(-1),oracle[j])
    parser=[]
    for r in rows:
        accepted=[]
        for p in PATCHES:
            try:
                compile(r["head"]+p+r["tail"],"<static parser>","exec")
                accepted.append(p)
            except SyntaxError:
                pass
        parser.append(accepted)
    parse_solves=sum(bool(p) and p[0]==PATCHES[g] for p,g in zip(parser,gold))
    with np.load(R/"logits.npz") as z:
        np.testing.assert_array_equal(gold,z["gold"])
        preds={k:z[k].argmax(-1) for k in z.files if k not in ("gold","patch_token_ids")}
        assert all(np.isfinite(z[k]).all() for k in preds)
        nll={}
        for k in preds:
            raw=z[k].astype("float64")
            centered=raw-raw.max(-1,keepdims=True)
            nll[k]=np.log(np.exp(centered).sum(-1))-centered[:,np.arange(n),gold]
    ok={k:oracle[np.arange(n),p] for k,p in preds.items()}
    native,donor,pre,on,off=(ok[k] for k in ("frozen_native","donor","frozen_coder","trained_coder","coder_on_knockout"))
    delta=lambda a,b:a.astype(float)-b.astype(float)
    did=delta(on,off)-delta(pre,ok["frozen_knockout"])
    intro=on & ~native
    lost=intro & ~off
    retained=intro & off
    strict=lost & donor & ~pre & ~ok["trained_native"]
    strict_controls=strict & ~ok["trained_shuffled"] & ~ok["trained_general"] & ~ok["trained_direct"]
    masks={"new":intro,"new_lost_when_off":lost,"new_still_solved_off":retained,
           "new_vs_frozen_graft":on&~pre,"new_vs_both_untouched_host_and_frozen_graft":intro&~pre,
           "loss_on_withdrawal":on&~off,"gained_on_withdrawal":off&~on,
           "donor_supported_new_lost":lost&donor,"strict_new_lost":strict,"strict_all_controls_fail":strict_controls}
    groups={k:{str(s):keys[mask[i]].tolist() for i,s in enumerate(SEEDS)} for k,mask in masks.items()}
    for k,mask in masks.items():
        groups[k]["all_seeds"]=keys[mask.all(0)].tolist()
        groups[k]["any_seed"]=keys[mask.any(0)].tolist()
        groups[k]["majority"]=keys[mask.sum(0)>=2].tolist()
    write("solve-ids.json",groups)
    ledger=[]
    for j,r in enumerate(rows):
        ledger.append(dict(id=r["id"],required=PATCHES[gold[j]],template=[int(gold[j]),r["style"]],
                          predicted={k:[PATCHES[int(v)] for v in p[:,j]] for k,p in preds.items()},
                          lost_when_off_seeds=[s for i,s in enumerate(SEEDS) if lost[i,j]],
                          strictly_transferred_seeds=[s for i,s in enumerate(SEEDS) if strict[i,j]]))
    write("solve-ledger.json",ledger)
    comparisons={"training_gain_coder":paired(delta(on,pre)),
                 "withdrawal_before":paired(delta(pre,ok["frozen_knockout"])),
                 "withdrawal_after":paired(delta(on,off)),
                 "change_in_reliance":paired(did),
                 "donor_vs_host":paired(delta(donor,native))}
    for a in ("native","shuffled","general","direct","knockout"):
        comparisons["coder_vs_trained_"+a]=paired(delta(on,ok["trained_"+a]))
    for a in ("general","shuffled"):
        comparisons["matching_"+a]=paired((delta(on,ok[a+"_on_coder"])+delta(ok["trained_"+a],ok["coder_on_"+a]))/2)
    wins=int((donor&~native).sum())
    losses=int((native&~donor).sum())
    counts={k:v.sum(1).tolist() for k,v in ok.items()}
    write("summary.json",dict(n=n,seed_order=SEEDS,counts=counts,standalone_discordant=[wins,losses],
          standalone_exact_p=exact_p(wins,losses),comparisons=comparisons,
          ledger_counts={k:{j:len(x) for j,x in d.items()} for k,d in groups.items()},
          exploratory_confidence=dict(task_nll={k:v.mean(1).tolist() for k,v in nll.items()},
                                      withdrawal_nll=paired(nll["coder_on_knockout"]-nll["trained_coder"])),
          parser_solves=parse_solves,unique_parse=sum(len(x)==1 for x in parser),
          unique_prompts_and_sources=len(seen[0]),
          by_token={PATCHES[g]:{k:v[:,gold==g].sum(1).tolist() for k,v in ok.items()} for g in range(4)}))
    print("Verified executable repairs, 12 concealed cases per item, distinct sources, predictions and paired contrasts.")


if __name__=="__main__":
    main()
