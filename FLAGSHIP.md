# Can a recipient model compute with a donor’s value?

## Stored facts transferred to a frozen 7B model

The best rank-256 map let frozen Qwen2.5-7B answer **11,604 of 20,000 direct questions (58.02%)** about people the map never saw during fitting. The 0.5B donor had learned randomly assigned facts about those people. Feeding a different person’s state through the same map reduced direct accuracy to the reported 3.75%, near the 3.73% chance level. Both models stayed frozen while the map trained.

| Condition | Direct facts, out of 20,000 | Country questions, out of 6,000 |
|---|---:|---:|
| Rank-256 map, best direct result | 11,604 | 1,247 |
| Donor alone | 19,715 | 597 |
| Frozen recipient alone | 775 | 269 |
| Donor answer relayed to the recipient as text | 19,472 | 1,386 |
| Recipient LoRA trained on the same map-fitting people | 702 | 691 |
| Rank-64 map using the untrained donor | 774 | 501 |

The best run recovered 58.86% of the donor’s direct accuracy, below the registered 60% threshold. Direct accuracy averaged 55.30% over three rank-256 seeds. The text relay led on direct and country questions. Evaluation scored the first answer token from a fixed set. The study selected the layer and rank on this evaluation panel, so the strongest number is a configuration-selection result.

The CPU audit recomputes the table and seed statistics from the retained per-question scores. The wrong-person result was saved as a summary and checked against the corresponding configuration record. The same study encompasses the operand interventions below and the approved stored-fact evaluation. [Registered E1 protocol](../research-program/protocols/E1.md) · [Audit](flagship/knowledge-7b-audit.json) · [Verifier](flagship/verify_knowledge.py)

## Held-out loop result

**The seven-coordinate edit changed both loop outputs and preserved the audit value in 359 of 501 native-qualified pairs (71.7%).** Qwen2.5-72B received the alternate operand from the 7B donor at one assignment site. The calibration examples used direct returns. The loop experiment required both swap directions to produce the exact tuple, including the unedited second field.

| Intervention | Native-qualified pairs, out of 501 | All pairs, out of 768 |
|---|---:|---:|
| Seven-coordinate donor edit | 359 | 430 |
| Dense linear stitch | 353 | 427 |
| Full recipient activation swap | 367 | 429 |
| Wrong donor value | 0 | 0 |
| Equal-norm orthogonal edit | 0 | 0 |

The donor edit cleared the registered transfer gate. It led the dense stitch by 1.20 percentage points, with a paired 95% interval from −1.60 to +3.99 points. The comparison provides evidence for selective operand transfer on this loop template within the Qwen family. It leaves the practical case for seven coordinates and generalization to other model families open. The conditional benchmark stopped at native qualification.

The Studio run took 18 minutes 34 seconds and peaked at 146.8 GB of MLX allocation. The independent verifier takes the pinned recipient tokenizer. It recalculates the answers, retokenizes the prompts and expected completions, checks the saved greedy traces and edit norms, and reproduces the paired statistics on CPU. [Scoring evidence and receipt](flagship/held-out-loop) · [Verifier](flagship/verify_loop.py) · [Pinned recipient tokenizer](https://huggingface.co/Qwen/Qwen2.5-72B/blob/efba10c8e54e91e0d9570ab5f7b51a958474d4cb/tokenizer.json)

## Earlier arithmetic results

The strongest starting result is **502 of 504 paired arithmetic reversals in frozen Qwen2.5-72B**. Seven coordinates derived from Qwen2.5-Coder-7B carried a changed operand. On a separate addition panel, two operand edits changed all four complete answers in 507 of 509 groups. Both panels used related Qwen checkpoints and known assignment sites. [One operand](nested_surgery/numeric_width_study/README.md) · [Two operands](nested_surgery/numeric_carry_72b_study/README.md)

## Studio qualification

The unedited recipient completed both variants in **43 of 64 loop groups** and **0 of 64 conditional groups**. In the conditional screen, 86 of 128 completions diverged before the end of the tuple fields. The original gate required 32 groups in each family, so the two-family run ended at qualification. The qualifying loop went forward under the separate registration below. The conditional remains untested under intervention. [Screen traces and receipt](flagship/qualification)

## What would change the conclusion

Freeze the existing seven-coordinate interface. Give the donor a fresh input digit in a direct-return program. Its hidden state supplies that digit to the intervention. Give the recipient a different starting digit and a Python function. The recipient must use the donor’s digit in either a three-way conditional or an accumulator loop, then complete an exact assertion containing the new result and an untouched audit value. The donor only sees its own program. Each pair tests the two directions of a swap, using the corresponding donor value each time.

Chen et al. already transferred steering directions between models with affine residual-stream maps and tested response language and instruction following. Cárdenas Piepereit injected projected states across model families and tested source retrieval from generated text. Bansal et al.’s CALM borrows coding and language capabilities through learned cross-attention between frozen models. Here the test is exact, paired execution on new program structures with a frozen interface, including preservation of an independent output. [Chen et al., NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/4569a868e7aa891248832ec08445d071-Abstract-Conference.html) · [Cárdenas Piepereit, 2026 preprint](https://arxiv.org/abs/2608.16347) · [Bansal et al., 2024](https://arxiv.org/abs/2401.02412)

XBridge and POND already report cross-family communication from hidden states and compare with text. This study measures selective use of one operand in a controlled computation. A practical communication claim would also require a measured baseline that generates the digit with the donor and relays it as text. [XBridge, 2026](https://arxiv.org/abs/2608.11676) · [POND, 2026 preprint](https://arxiv.org/abs/2611.21301)

## Registered comparison

Use the pinned BF16 Qwen2.5-Coder-7B and Qwen2.5-72B checkpoints and the archived 7B-to-14B and 14B-to-72B matrices at completed blocks 6 and 17. The rank, sites and gains are fixed. Fit a dense affine-stitch comparison on the same 256 direct-return contrasts used to align to 72B, with the 7B donor as input. Use ridge with penalty 0.02 times the mean diagonal of the calibration Gram matrix. Use donor partner-minus-current contrasts and match the dose of the seven-coordinate edit. Calibration uses only activations from direct-return examples. The evaluation uses separate seeds and changes the function structure after the patched site.

| Condition | What it tests |
|---|---|
| Unedited and recipient given the correct digit in text | Baseline and native qualification on both variants |
| Frozen seven-coordinate donor edit | Whether the intended computation and audit value both come out right |
| Wrong donor digit | Whether success depends on the specific content carried by the edit |
| Norm-matched orthogonal edit | Whether moving the activation by the same amount is sufficient |
| Dense ridge stitch | How the narrow interface compares with a whole-width projection |
| Swap the whole recipient activation at the same site | Whether that site admits the desired intervention within the recipient |

Run the same comparison in two task families. Generate 64 native-screen groups per family, then a separate 768 evaluation groups per family. Each group has distinct native, partner and wrong-donor results and a fixed independent audit value. Use the fraction of pairs with both complete counterfactual tuples correct as the primary outcome. All tokens after the supplied opening parenthesis must match unique greedy vocabulary maxima, including a valid closing parenthesis. The canonical tuple formatting is fixed, and the Python oracle validates the exact mathematical outputs. Report total groups and native-qualified groups, outcomes on both denominators and 95% paired bootstrap intervals using the group as the sampling unit. Report dose, wall time and peak MLX memory.

Stop before the intervention panel if fewer than 32 of 64 native screen groups qualify in either family. The recipient needs at least 384 native-qualified groups and a 70% whole-state success rate in each family for a negative comparison to be interpretable. The targeted result passes if it reaches 70% joint exact success in each family, its advantage over both the wrong donor and orthogonal edits reaches 20 percentage points in each with paired intervals above zero, and the native qualification floor is met. Claim an advantage over the dense stitch only if the paired lower bound is positive in both families and the point gain is at least five points in both. A dense-stitch tie or win still allows an exact-transfer result, with the relative performance reported. If the positive control passes and the edit fails, report where generalization stopped and end this comparison. A screen or positive-control failure is an inconclusive test of transfer.

Register one job with no successor. Budget up to three hours of intervention scoring, based on the earlier 72B whole-answer run, with sequential model loading, a 2 GiB MLX cache and no new checkpoints. A timed-out run keeps its partial artifacts and reports an incomplete outcome. If the first comparison supports broadening, the next decision is whether to preregister a second recipient from an independently trained family. Consolidate the evidence and limitations in one study before submitting it to Pubroot.

## Loop comparison registered after native qualification

The two-family protocol ended at qualification. The loop met its native screen threshold independently, with 43 of 64 paired solves. The 768 separate loop evaluation groups specified above have had no model scoring. Run that panel once to test whether the frozen donor interface transfers exact counterfactual outputs on a loop, including an unchanged second field. The recipient can compute the right answer by any mechanism. Leave the conditional panel closed and retain its result alongside the loop evidence.

Keep the checkpoints, matrices, dose, seeds and six arms specified above fixed. Use the original loop donor seed of 280000. Check the prior native screen against its pinned Studio receipt before scoring any interventions. Apply the original outcome thresholds to this one family. First require at least 384 native-qualified groups and 70% success under whole-state exchange. Given that positive control, a passing seven-coordinate edit extends the arithmetic result to a loop. A miss bounds the interface to the evidence already collected. A dense-stitch tie or win limits the case for the seven-coordinate design, even if transfer passes. The budget remains three hours and this job has no automatic successor.
