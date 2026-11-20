"""Summarize where the output-only lookup works using saved panel choices."""
import json
import numpy as np
from experiment import R,output_key,rows_for,write

def main():
    train=rows_for("train")
    keys={output_key(r["prompt"]) for r in train}
    out={}
    for split in ("qualification","confirmation"):
        rows=json.loads((R/f"{split}-problems.json").read_text())
        visible=[output_key(r["prompt"]) for r in rows]
        with np.load(R/f"{split}.npz") as z:
            gold=z["gold"];pred=z["shortcut"].argmax(1);correct=pred==gold
        exact=np.array([k in keys for k in visible])
        constant=np.array([k==(1,1,1,1) for k in visible])
        out[split]=dict(correct_by_depth=[int(correct[gold==j].sum()) for j in range(4)],
            count_by_depth=np.bincount(gold,minlength=4).tolist(),ones_gold_counts=np.bincount(gold[constant],minlength=4).tolist(),
            keys_previously_seen=int(exact.sum()),correct_on_seen=int(correct[exact].sum()),
            keys_unseen=int((~exact).sum()),correct_on_unseen=int(correct[~exact].sum()))
    write(R/"shortcut-analysis.json",out)
    print(out)
if __name__=="__main__":main()

