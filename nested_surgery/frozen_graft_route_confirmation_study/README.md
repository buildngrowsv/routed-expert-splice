# A fixed confidence route added 106 repairs while calling the 14B graft on 23% of prompts

On **2,304** new Python repair groups with misleading brackets, confidence routing solved **1,481** groups versus **1,375** for the intact 7B specialist. It requested the 14B graft on **23.05%** of prompts and beat random calls at the identical budget. The matched neutral layout scored **1,956** versus **1,960** standalone. Its paired lower confidence bound was −1.17 percentage points, within the prespecified two-point tolerance. The frozen rule met every registered criterion in this confirmation.

![Complete repair groups for standalone, matched random, confidence routing and always-blend policies](route.png)

## Keep the routing rule fixed and expand the sample

The earlier prospective panel used 768 triplets per layout. It found a decoy gain of 61 repairs at 22.92% graft calls and a neutral difference of −0.78 points with a 95% lower bound of −2.60. The neutral result left the overall decision unresolved. This run held both pretrained models, the existing 168,960-parameter affine bridge, every routing parameter and all numerical gates fixed. It evaluated 2,304 new matched triplets per layout, with IDs 87000–89303 disjoint from the calibration and earlier evaluation sets. [Previous prospective result](../frozen_graft_route_study/README.md) · [Preregistered confirmation](../FROZEN_GRAFT_ROUTE_CONFIRMATION.md)

Each triplet has three versions of the same Python repair. Changing its opening delimiter changes the sole valid closing delimiter among four candidates. The decoy layout inserts misleading brackets in an inert string and a comment. The neutral layout replaces the same characters with letters. Whole-group success requires all three highest-scoring candidates to be correct. The frozen interface takes 32 features from the intact Qwen2.5-Coder-7B's final feed-forward module and adds their learned translation to the 14B residual stream after block 27. The rule measures the margin between the two highest 7B logits. Only when that margin is at most **0.5625** does it blend 75% of the 7B logits and 25% of the graft's. This margin was selected on a separate archived panel before both prospective evaluations. No trainable weights were fit on the new group IDs. [Frozen bridge](../learned_output_port_study/README.md) · [Frozen decoy test](../learned_output_port_decoys_study/README.md) · [Guo et al. on post-hoc confidence calibration](https://arxiv.org/abs/1706.04599)

## Paired controls and cost

| Policy | Neutral complete / 2,304 | Decoy complete / 2,304 |
|---|---:|---:|
| Intact 7B specialist | 1,960 | 1,375 |
| Always use the 14B graft | 899 | 673 |
| Hard substitute on the confidence mask | 1,871 | 1,327 |
| Random blend at the same call count | 1,958.88 | 1,404.41 |
| Confidence-gated blend | 1,956 | 1,481 |
| Always blend | 1,952 | 1,498 |

Confidence routing gained **4.60 points** over standalone on decoys (95% paired interval +3.13 to +6.03), with 204 recovered and 98 lost groups. Against the mean of 32 seeded equal-budget random routes, it gained **3.32 points** (+2.18 to +4.47). On neutral prompts it gained 63 groups and lost 67, a net **−0.17 points** (−1.17 to +0.74). The rule called the graft on 1,593 of 6,912 decoy prompts and 825 of 6,912 neutral prompts. On decoys it retained 106 of the net 123 groups gained by constant blending. The additional compute relative to standalone is the selected 14B host pass. [Full logits and masks, neutral](results/neutral.npz) · [Full logits and masks, decoys](results/decoys.npz) · [Independent audit](audit.json)

Exhaustive offline scoring supplies the paired comparator logits. A separate check ran the host only on selected prompts from eight input batches per layout. The resulting candidate winners agreed on all nine neutral and 17 decoy calls. This verifies the sparse execution path on those inputs and records the full-panel request fraction. It leaves measured end-to-end latency and exact sparse parity on the rest of the panel open. The results concern executable artificial delimiter repairs and one pair of Qwen checkpoints. An ordinary parser solves the same benchmark exactly.

## Reproduction

The frozen BF16 models are Qwen2.5-Coder-7B revision `0396a76181e127dfc13e5c5ec48a8cee09938b02` and Qwen2.5-14B revision `97e1e76335b7017d8f67c08a19d103c0504298c9`. The Apple M5 Ultra Studio, with 256 GiB unified memory, ran macOS build 26B118, Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and tokenizers 0.23.3. Exhaustive evaluation took 498.39 seconds and peaked at 58.47 GB of MLX allocation. Verify the hashes, 6,912 prompts per layout, matched route counts, 5,000 paired group resamples and all acceptance gates using the command below. For model replay, run the pinned script with `--confirmation` after linking the public trained-bridge result directory into a fresh queue root as `runs/large-qwen-learned-output-port` and setting `STUDIO_QUEUE_ROOT` and `STUDIO_JOB_DIR`. The owner proposed the expert and routing research. OpenAI Codex registered, implemented, ran, audited and documented the confirmation. [Pinned runner](https://github.com/buildngrowsv/routed-expert-splice/blob/17fcd359c0397b0f3ca02d067a84cedb066dbc8a/nested_surgery/frozen_graft_route.py) · [Receipt](results/receipt.json) · [Environment](results/studio-environment.txt)

```sh
uv run --with numpy python nested_surgery/frozen_graft_route_study/verify.py --confirmation
```

[Published Pubroot article](https://pubroot.com/ai/fine-tuning/a-fixed-confidence-route-added-106-repairs-while-calling-the-14b-graft-on-23-of-prompts/) · [Public review](https://github.com/buildngrowsv/pubroot-website/issues/559)
