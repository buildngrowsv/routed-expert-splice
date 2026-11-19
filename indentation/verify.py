"""Execute all four candidate programs against the released assertions and hidden cases."""
import ast
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent/"results"

def solve(row):
    prefix,tail=row["prompt"].removeprefix("<|fim_prefix|>").removesuffix("<|fim_middle|>").split("<|fim_suffix|>")
    outputs=[];public=[]
    for width in (4,8,12,16):
        program=ast.parse(prefix+" "*width+tail)
        function,*checks=program.body
        env={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),"<indent-benchmark>","exec"),env)
        f=env["tally"]
        visible=[]
        for c in checks:
            assert isinstance(c,ast.Assert)
            comparison=c.test
            assert isinstance(comparison,ast.Compare) and isinstance(comparison.ops[0],ast.Eq)
            inputs=ast.literal_eval(comparison.left.args[0])
            visible.append(f(inputs)==ast.literal_eval(comparison.comparators[0]))
        assert len(visible)==4
        public.append(all(visible))
        outputs.append([f(x) for x in row["hidden"]]==row["expected"])
    assert sum(outputs)==sum(public)==1 and outputs==public
    return outputs.index(True)

def main():
    rows=json.loads((R/"problems.json").read_text())
    r=json.loads((R/"discovery.json").read_text())
    gold=np.array([solve(z) for z in rows])
    assert np.array_equal(gold,np.tile(np.arange(4),64))
    analysis={}
    with np.load(R/"discovery.npz") as data:
        np.testing.assert_array_equal(gold,data["gold"])
        success={}
        for name in ("host","donor"):
            picks=data[name].argmax(1)
            hit=(picks==gold);success[name]=hit
            assert hit.sum()==r[name]
            quarters=picks.reshape(-1,4)
            constant=(quarters==quarters[:,0,None]).all(1)
            analysis[name]=dict(prediction_counts=np.bincount(picks,minlength=4).tolist(),
                                quartets_unchanged=int(constant.sum()),quartets_both_or_more=int((hit.reshape(-1,4).sum(1)>=2).sum()))
        assert int((success["host"]&~success["donor"]).sum())==r["host_only"]
        assert int((success["donor"]&~success["host"]).sum())==r["donor_only"]
    assert r["discovery_qualified"]==(r["donor"]-r["host"]>=.05*len(rows))
    result=dict(unique_visible_and_hidden=256,legal_programs_executed=1024,choice_analysis=analysis)
    (R/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)
if __name__=="__main__": main()

