# A narrow 7B edit sometimes displaced a saved value in a 72B overwrite control

Editing a later assignment in frozen Qwen2.5-72B switched all current-value answers in **145 of 146** native-qualified groups with either the seven-dimensional donor or a context-matched whole-state exchange. In the saved-value condition, the donor caused the edited overwrite to appear as the answer in four groups. The matched whole-state and orthogonal arms caused zero such intrusions. The 2.74-point donor excess missed the registered three-point effect threshold. Exact saved-answer preservation was 137 groups with the donor and 138 with the whole-state exchange.

![Current-value switching, saved-value preservation and edited-overwrite intrusions](overwrite.png)

## Keep the saved value fixed during a whole-state exchange

The earlier saved-value benchmark observed 14 wrong replies after a transplant at an irrelevant overwrite. In each reply the top answer was the edited overwrite plus the signed offset. That benchmark exchanged the whole state only at the assignment the function read. This comparison applies both edits at the same overwrite location, using a source prompt with an identical prefix through the preceding token and the same earlier saved value. [Earlier saved-value result](../numeric_snapshot_72b_study/README.md) · [Registered matched comparison](../NUMERIC_OVERWRITE_CONTROLS_72B.md)

Each group crosses two original values with two later overwrite values. A saved and a current return expression use the same signed offset. The resulting group has eight prompts. Two prompts at a time differ only at the later integer token. Running the Python functions confirms the same returned value under an overwrite change when the expression reads the saved binding, and a changed returned value when it reads the current binding. The donor coordinates come from a separate function that returns the intended replacement digit. [Archived host and donor programs](results/programs.json)

Both Qwen2.5-Coder-7B and Qwen2.5-72B checkpoints are frozen BF16. Use the archived donor alignment from the state after six donor blocks and change the overwrite token after 17 host blocks. For host vector `h` and seven translated coordinates `c`, the narrow replacement is `h' = h + (c − ((h − μ)R + ν)) Wᵀ`, where the archived read and write matrices are `R` and `W`, and `μ` and `ν` are calibration centers. The whole-state exchange copies all 8,192 coordinates at that token from its paired counterpart. The orthogonal edit matches the donor's Euclidean intervention size in float32 before casting. The rest of the host continues from the same computation boundary. [Pinned runner](https://github.com/buildngrowsv/routed-expert-splice/blob/b5a8e086a68aeac4808d9bc14d9c79a722d0ac6c/nested_surgery/identifier_snapshot_overwrite_72b.py) · [Donor map](../numeric_absolute_study/results/map.npz) · [72B alignment](../numeric_width_study/72b/alignment.npz)

## Group-level outcomes

The unchanged host solved all eight prompts in 43 of 64 qualification groups and in 146 of 256 disjoint test groups. The table conditions on the latter 146. A switch or preservation requires each predicted token to rank uniquely first over the whole vocabulary. An intrusion requires at least one lost saved answer whose top token equals the counterfactual overwrite plus the offset.

| Edit at the later assignment | All current-value answers switch | All saved-value answers preserved | At least one edited-overwrite intrusion |
|---|---:|---:|---:|
| Seven-dimensional donor | 145/146 | 137/146 | 4/146 |
| Context-matched full-state exchange | 145/146 | 138/146 | 0/146 |
| Equal-norm orthogonal | 0/146 | 143/146 | 0/146 |

The donor–whole difference in exact intrusions was +2.74 points. The registered 5,000-draw paired bootstrap interval was +0.68 to +5.48 points. A post hoc two-sided exact paired comparison of the four discordant groups yields `p = .125`, reflecting how few events distinguish the two arms. The registered result remains unresolved. The two edits had similar overall saved-answer loss. The rare overwrite-specific error was observed only with the narrow edit in this panel. These controlled Python expressions test a known assignment site and one previously fitted representation. [Raw whole-vocabulary ranks and norms](results/overwrite.npz) · [Computed results and uncertainty](audit.json)

Run `uv run --with numpy python nested_surgery/numeric_overwrite_72b_study/verify.py` from the repository root to execute the prompt cases, check the frozen map and alignment hashes, and reproduce all paired endpoints. The Studio run took 1,135 seconds and reached 147.24 GB peak MLX allocation. Hardware was Apple M5 Ultra with 80 GPU cores and 256 GiB unified memory. It ran macOS 27.2 (26B118), Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5, and tokenizers 0.23.3. The owner proposed the fixed-reference research questions. OpenAI Codex implemented, ran, audited and wrote up the paired comparison. [Execution receipt](results/receipt.json) · [Independent verifier](verify.py)

[Pubroot submission and review](https://github.com/buildngrowsv/pubroot-website/issues/514)
