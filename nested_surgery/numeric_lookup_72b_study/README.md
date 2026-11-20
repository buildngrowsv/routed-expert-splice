# A transplanted key redirected about half of paired 72B dictionary lookups

A frozen Qwen2.5-72B model followed a transplanted 7B key to the counterfactual answer in both mirrored dictionary contexts in **92 of 183** native-qualified groups. An equal-norm orthogonal edit switched **zero** and a full hidden-state exchange at the same declaration switched **96 of 183**. The donor preserved **182 of 183** groups where both keys led to the same value. Both active edits missed the preregistered 80% switching requirement.

![Indexed-routing switch and answer-preservation outcomes](lookup.png)

## Use the transplanted number as a dictionary key

Each generated function assigns a single-digit key, makes a two-entry dictionary, and returns the value at that key plus a signed offset. Each paired prompt differs only at the key assignment. One context has the original table, one swaps the table values, and one assigns the same value to both keys. Mirroring requires the same key substitution to move the output in opposite directions. The constant table checks that the key substitution can preserve the answer when both keys have the same effect. The answer digits appear nowhere as numeric literals in the generated programs. All expected answers are checked by executing the archived Python. [Registered dictionary protocol](../NUMERIC_LOOKUP_72B.md) · [Full host and donor panel](results/programs.json)

The preceding conditional-statement task cleared only 26 of 64 native qualification groups and stopped. The dictionary task qualified 51 of 64 independent groups and completed all six answers in 183 of 256 new test groups. On these groups every intervention uses both variants in all three contexts. A prediction must rank uniquely first over the entire vocabulary. [Earlier native screen](../numeric_select_72b_study/branch_screen/README.md) · [Dictionary qualification ranks](results/screen.npz)

## Compare edits at the same token boundary

The frozen BF16 checkpoints are Qwen2.5-Coder-7B and Qwen2.5-72B. Donor coordinates come from a separate single-variable program after six blocks and are passed through the previously fitted affine map. At the target key's declaration after 17 recipient blocks, the narrow edit uses `h' = h + (c − ((h − μ)R + ν))Wᵀ`. Here `h` is the 8,192-value recipient state, `c` is the seven-coordinate donor target, and the archived `R, W, μ, ν` define the fixed read and write interface. The orthogonal control uses the same change size and is computed in the same pre-cast precision. The full-state exchange replaces the whole declaration vector with its paired counterpart. The recipient is causal and the key is assigned before the dictionary, so the key state has the same prefix and donor coordinate target across the original, mirrored and constant cases. Nothing is trained or recalibrated. [Pinned implementation](https://github.com/buildngrowsv/routed-expert-splice/blob/ddcca579733367d0282894afd3dc752b47a74bc0/nested_surgery/identifier_numeric_route_72b.py) · [Frozen donor map](../numeric_absolute_study/results/map.npz) · [Frozen host alignment](../numeric_width_study/72b/alignment.npz)

| Intervention | Original table switches | Mirrored table switches | Both switch | Constant table preserves | All criteria for group |
|---|---:|---:|---:|---:|---:|
| Narrow donor | 122/183 | 118/183 | 92/183 | 182/183 | 92/183 |
| Equal-norm orthogonal | 0/183 | 0/183 | 0/183 | 183/183 | 0/183 |
| Full-state exchange | 125/183 | 125/183 | 96/183 | 181/183 | 95/183 |

The joint donor advantage over orthogonal was 50.27 points (95% paired interval 43.17 to 57.38). The 80% switching gate required 147 of the 183 native-qualified groups. The donor and full-state swap switched both mirrored contexts in the same 89 groups. A post hoc paired narrow-minus-whole switching difference was −2.19 points (−5.46 to +1.09). Those data show address-dependent changes in a substantial subset and limit the reliability claim at this declaration site. The study does not localize where the unaffected cases recover the original answer or establish that the model executes general lookup code. [Raw ranks and edit norms](results/selector.npz) · [Paired outcome and overlap audit](audit.json)

Run `uv run --with numpy python nested_surgery/numeric_lookup_72b_study/verify.py` from the repository root to check the prompt pairs by execution, compare the frozen map and alignment hashes, and recompute the rank endpoints. The Studio took 460 seconds at 146.7 GB peak MLX allocation. Hardware was Apple M5 Ultra with 80 GPU cores and 256 GiB unified memory. The run used macOS 27.2 (26B118), Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and tokenizers 0.23.3. The owner proposed investigating reused model parts and conditional routing. OpenAI Codex implemented the test, ran it, verified the evidence and assisted with the writing. [Execution receipt](results/receipt.json) · [Independent verifier](verify.py)
