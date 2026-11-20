# A dictionary-only attention mask improved early key steering and missed the registered target

Blocking the dictionary tokens from reading an early key before block 17 raised mirrored whole-state switching from **101 to 123 of 157** native-shared groups in frozen Qwen2.5-72B. An equal-size mask on the preceding syntax token switched **102**. The target mask gained 13.38 points over that syntax control (95% paired interval +5.10 to +21.66). The observed 78.34% switching rate and the paired effect size both missed the registered thresholds.

![Native coverage and shared-cohort switches under the dictionary-only masks](dictionary-mask.png)

## Restrict the attention intervention to the dictionary line

An earlier study removed access to the key for every later token through the first 17 blocks. Native coverage dropped from 173 to 17 groups. This follow-up uses the same early-key programs and restricts the blocked queries to the 15 tokens of the dictionary assignment. The exact tokenizer offsets are archived. Every other token, including the final lookup expression, keeps its usual causal access. The key mask removes attention from each table token to the original numeral. The matched syntax mask removes the same edges to the immediately preceding nonnumeric assignment token. Ordinary causal attention resumes after the full key-state exchange. [Broader mask outcome](../numeric_lookup_preedit_attention_72b_study/README.md) · [Registered dictionary-only comparison](../NUMERIC_LOOKUP_PREEDIT_ATTENTION_72B.md) · [Pinned implementation](https://github.com/buildngrowsv/routed-expert-splice/blob/d29e00dde9be0f7ea6d08a5e427c67bfd9e834ce/nested_surgery/identifier_lookup_preedit_attention_72b.py)

The 256 groups reuse the fresh independent lookup panel. In each group, the same two keys choose opposite answers under mirrored tables and the same answer under a constant table. Every prompt has an executable Python answer. After 17 frozen recipient blocks, exchange the 8,192-value key state with the paired prompt's state in the same masked condition. Both variants differ only at that numeral. The unmasked arm uses its previously archived whole-state and native scores. The comparison uses no donor fitting. Each expected answer must rank uniquely first across the full vocabulary. [Exact source prompts](../numeric_lookup_fresh_72b_study/results/programs.json) · [Unrestricted ranks](../numeric_lookup_fresh_72b_study/results/early.npz)

| Pre-edit condition | Native correct, all six | Both mirrored lookups switch | Constant-table answers preserved |
|---|---:|---:|---:|
| Unrestricted | 173/256 | 101/157 | 156/157 |
| Dictionary-to-syntax mask | 176/256 | 102/157 | 155/157 |
| Dictionary-to-key mask | 159/256 | 123/157 | 154/157 |

The intervention outcomes condition on the groups solved natively in all three arms. The target mask gained 14.01 points over unrestricted attention (paired 95% interval +5.73 to +22.29). It was three groups short of 80% mirrored switching. It also fell below the 15-point event-rate gain required against both controls. Constant-table preservation met its 95% criterion. The result is consistent with a contribution from direct key reads by dictionary tokens under this specific attention cut. Other tokens can still relay information and the comparison leaves the larger placement gain open. [Masked ranks and table spans](results/attention.npz) · [Paired audit](audit.json) · [Late alias follow-up](../NUMERIC_LOOKUP_ALIAS_72B.md)

Run `uv run --with numpy python nested_surgery/numeric_lookup_table_attention_72b_study/verify.py` to execute archived program assertions, verify the source hashes and span records, and recompute the counts and paired intervals. The Studio run took 496 seconds, reaching 146.71 GB peak MLX allocation on Apple M5 Ultra with 80 GPU cores and 256 GiB unified memory. The environment was macOS build 26B118, Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and tokenizers 0.23.3. The owner proposed research on walls and conditional routing. OpenAI Codex implemented, ran, audited and documented the comparison. [Execution receipt](results/receipt.json) · [Independent verifier](verify.py)

[Pubroot submission and review](https://github.com/buildngrowsv/pubroot-website/issues/536)
