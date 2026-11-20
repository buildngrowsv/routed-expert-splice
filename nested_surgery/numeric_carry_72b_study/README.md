# Two donor operands changed complete answers in 507 of 509 groups at 72B

A frozen 72B model used two independently supplied 7B activations to change all four complete arithmetic answers in **507 of 509** eligible groups. A two-site full-state swap also solved **507**, and matched orthogonal edits solved zero. The experiment reused the seven-coordinate alignment fitted on single-operand examples without recalibrating it. The model correctly changed 742 of 743 eligible answers from one digit to two and all 743 in the other direction.

![Complete results in the 32B and 72B recipients and the two directions of answer-length change](carry-72b.png)

## Same donor bridge at two assignments

Each group contains four Python functions formed by pairing two possible left operands with two possible right operands. All four native sums differ and at least one crosses ten. A donor declaration for either digit comes from a distinct 7B source function. The native 72B recipient solved all 128 qualification groups. Test prompts are the same 512 archived groups used earlier at 14B and 32B. The 72B test natively solved all four variants in 509 groups. [Registered design](../NUMERIC_CARRY_72B.md) · [Archived host and donor functions](../numeric_carry_study/results/programs.json)

| Role | Frozen BF16 checkpoint | Width | Completed blocks |
|---|---|---:|---:|
| Donor | Qwen2.5-Coder-7B [0396a761](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/tree/0396a76181e127dfc13e5c5ec48a8cee09938b02) | 3,584 | 6 |
| Original affine map | Qwen2.5-14B [97e1e763](https://huggingface.co/Qwen/Qwen2.5-14B/tree/97e1e76335b7017d8f67c08a19d103c0504298c9) | 5,120 | 10 |
| Earlier recipient | Qwen2.5-32B [1818d358](https://huggingface.co/Qwen/Qwen2.5-32B/tree/1818d35814b8319459f4bd55ed1ac8709630f003) | 5,120 | 13 |
| Current recipient | Qwen2.5-72B [efba10c8](https://huggingface.co/Qwen/Qwen2.5-72B/tree/efba10c8e54e91e0d9570ab5f7b51a958474d4cb) | 8,192 | 17 |

For donor activation `d`, the original 7B-to-14B map proposes coordinates `c = d A F + b`. The 72B assignment activation `h` is centered and read into those coordinates by a pseudoinverse of the fitted 8,192×7 write matrix `W`. The applied edit is `h' = h + (c − ((h − μ)R + ν)) Wᵀ`, where `R` is the 8,192×7 read matrix and `μ, ν` are the archived calibration means. Apply that rule independently at both assignment positions after block 17. The rest of the state is unchanged at that boundary. The orthogonal control uses the same amount of pre-cast change at each position in a basis orthogonal to the fitted write directions. The full-state control replaces both complete assignment activations with partner values. Edits are computed in float32 and cast to BF16 for model execution. [Prior separate alignment fit](../numeric_width_study/README.md) · [Original matrices](../numeric_absolute_study/results/map.npz) · [Carry implementation](../identifier_numeric_carry_72b.py)

A completion must choose each expected digit uniquely over the full vocabulary and choose an accepted terminator after the last digit. The scoring code extends the expected prefix only to inspect its greedy path. A wrong earlier argmax invalidates the whole completion. A group succeeds only if it succeeds in all four prompt variants. [Greedy trace implementation](../identifier_numeric_carry.py)

## Matched outcomes

| Intervention | 32B, out of 509 shared native groups | 72B, out of 509 | 72B one to two digits, out of 743 | 72B two to one digit, out of 743 |
|---|---:|---:|---:|---:|
| Aligned donor edit | 356 | 507 | 742 | 743 |
| Equal-norm orthogonal edit | 0 | 0 | 0 | 0 |
| Swap both full native states | 451 | 507 | 743 | 742 |

The 72B narrow edit cleared the 80% success threshold and beat its orthogonal control by 99.61 percentage points. On the same 509 eligible groups, its paired gain over the earlier 32B implementation was **29.67 points**, 95% interval **25.54 to 33.60**. The 32B comparison uses the earlier single scalar gain and same-width axes. The 72B comparison uses the separately documented contrast-derived alignment. The recipient and alignment procedure both differ, and the bigger model also responded better to the whole-state swap. The result establishes two-input composition for this held-out addition format. It does not isolate a standalone effect of parameter count or establish general code-generation accuracy. [Archived 32B study](../numeric_carry_32b_study/README.md) · [Shared-outcome audit](audit.json)

The interventions extend earlier work on moving linear features across language models by measuring complete arithmetic carry using two distinct donor sources. [Chen et al., NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html)

Run `uv run --with numpy python nested_surgery/numeric_carry_72b_study/verify.py` to execute and verify the test programs, check the fitted interface and archives by digest, and recalculate the vocabulary-trace counts, per-site norm matching and paired statistics. The Studio run took 1,403 seconds and reached 147.35 GB peak MLX allocation. Hardware was an Apple M5 Ultra with 80 GPU cores, 256 GiB unified memory and Metal 4. The environment was macOS 27.2 (26B118), Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and tokenizers 0.23.3. The owner proposed larger-scale expert transplantation. OpenAI Codex implemented and ran the paired test, verified the evidence and assisted with writing. [Pinned Studio receipt and scores](results) · [Independent verifier](verify.py)
