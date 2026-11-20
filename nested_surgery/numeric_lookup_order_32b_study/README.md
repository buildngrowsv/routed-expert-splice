# Late dictionary key control replicated in a frozen 32B model

A whole-state key swap redirected both mirrored dictionary lookups in **151 of 151** groups that frozen Qwen2.5-32B solved natively in both orders when the key followed the table. With the key before the table, it redirected **84 of 151**. The paired gain was **44.37 percentage points** (95% interval +36.42 to +52.32). In the cohort solved in all three layouts, the late key with the exact early token gap redirected **149 of 150** groups. The unedited model solved fewer groups in the late order across the full panel.

![32B results for all three lookup layouts and the shared 32B/72B comparison](second-host.png)

## Reuse the input groups in a different frozen checkpoint

The 256 groups come from the prior independent 72B replication. Each group uses two selector values across mirrored dictionary mappings and a constant mapping. Every function returns the value for the selected key plus a signed offset. The early and late orders exchange only two definition lines. The padded late order inserts a Python comment after the key to restore the early key-to-answer token distance under the 32B tokenizer. Every expected answer stays the same. The prefix through the key is identical in the padded and unpadded late cases. [Preregistered 32B comparison](../NUMERIC_LOOKUP_ORDER_32B.md) · [Input programs and tokenization](results/programs.json) · [72B matched-gap result](../numeric_lookup_gap_72b_study/README.md)

For each order, score the frozen BF16 Qwen2.5-32B checkpoint without edits, and exchange the entire 5,120-value key state between prompts that differ only at that numeral. The edit follows block 13, the previously used intervention boundary. In all conditions the answers are scored by unique first rank across the full vocabulary. Switching requires both directions across both mirrored tables. Preservation requires both constant-table answers to remain correct. The test uses a whole-state exchange and no donor model. [Pinned runner](https://github.com/buildngrowsv/routed-expert-splice/blob/4d0348e77fafbb3a0a6c12015f20cc341197288f/nested_surgery/identifier_lookup_order_32b.py) · [Frozen 32B checkpoint](https://huggingface.co/Qwen/Qwen2.5-32B/tree/1818d35814b8319459f4bd55ed1ac8709630f003)

| Measure | Early key | Late key | Late key with matched gap |
|---|---:|---:|---:|
| Native correct groups on the full panel | 200/256 | 176/256 | 177/256 |
| Whole-state switching over the shared three-order cohort | 84/150 | 150/150 | 149/150 |
| Constant-table preservation over that cohort | 150/150 | 149/150 | 150/150 |

The primary two-order cohort included 151 groups. Late constant-table preservation was 150/151. The experiment cleared its preregistered thresholds. In the three-order cohort the padded-minus-early lift was +43.33 points (paired 95% interval +35.33 to +51.33). The fixed gap result concerns this comment control and known key location.

A post hoc audit of the 107 groups solved natively in all six model-and-order combinations found 32B whole-state switching of 62 early, 107 late and 107 padded. The comparable 72B counts were 66, 106 and 106. This descriptive overlap shows the pattern in the same examples in both pretrained hosts. The model widths and intervention boundaries differ, and both checkpoints come from the same Qwen family. [Early ranks](results/early.npz) · [Late ranks](results/late.npz) · [Matched-gap ranks](results/gap.npz) · [Independent audit](audit.json)

Run `uv run --with numpy python nested_surgery/numeric_lookup_order_32b_study/verify.py` to check the exact program pairs, execute assertions, and reproduce cohort sizes, scores and paired intervals. The Studio run took 317 seconds and reached 66.40 GB peak MLX allocation. It used Apple M5 Ultra with 80 GPU cores and 256 GiB unified memory, macOS build 26B118, Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and tokenizers 0.23.3. The owner proposed researching conditional routing and fixed references. OpenAI Codex implemented, ran, audited and documented the study. [Execution receipt](results/receipt.json) · [Verifier](verify.py)

[Published Pubroot article](https://pubroot.com/ai/nlp/late-dictionary-key-control-replicated-in-a-frozen-32b-model/) · [Pubroot submission and review](https://github.com/buildngrowsv/pubroot-website/issues/531) · [Attention intervention protocol](../NUMERIC_LOOKUP_PREEDIT_ATTENTION_72B.md)
