# Calibrating seven 14B operand coordinates in a new recipient

The same-width positive control for the 72B experiment is complete. In Qwen2.5-32B, alignment from matched input states reversed **426 of 507** qualified arithmetic pairs. Exchanging the full native assignment state also reversed 426. Orthogonal edits reversed zero. The previously released single-gain method reversed 434 of these same pairs. The wider 72B arm uses the same fitting rule and held-out prompts.

We reuse the seven-coordinate interface learned in the 14B model and the frozen 7B donor. The alignment sees 256 new input pairs. Each pair differs in its assigned digit and provides the full recipient-state difference at the known assignment position. Ridge regression fits seven writing directions to those differences. The read uses the pseudoinverse and centered recipient state to express a live activation in the original coordinate system. The ridge estimate changes direction and magnitude. No output answer participates in this 32B alignment. [Registered method and recipient control](../NUMERIC_WIDTH_72B.md) · [Alignment implementation](../identifier_numeric_width_72b.py) · [Prior single-gain calibration](../numeric_scalar_study/README.md)

| Edit in frozen 32B | Reversed pairs, out of 507 |
|---|---:|
| Matched-input alignment | 426 |
| Equal-norm orthogonal edit | 0 |
| Full native replacement | 426 |
| Prior same-axis single gain | 434 |

The alignment beat the orthogonal control by 84.02 points with a paired 95% interval of 80.87–86.98. The earlier result uses the original coordinate axes with a single global gain. This control instead learns a low-rank alignment, the method the wider recipient will require. These comparisons use one assignment site and the calibrated family of arithmetic prompts. [Audited counts and fit checks](audit-32b.json) · [Pinned Studio evidence](32b)

Verify the ridge solution, write-read identity, control orthogonality, archive digests, matched norms and full-vocabulary scores using `uv run --with numpy python nested_surgery/numeric_width_study/verify.py 32b`. The 32B run took 3 minutes 36 seconds and used 66.9 GB peak MLX allocation on the Apple M5 Ultra. [Prior affine model stitching](https://proceedings.nips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html)
