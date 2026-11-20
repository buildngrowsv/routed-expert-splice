# The same activation edit followed the class the 14B model was asked to read

A 25-dimensional edit to one Python declaration flipped **498 of 512** paired completions when the query selected the edited class. It flipped **1 of 512** when the query selected the other class. Each declaration received the identical edit in both roles. The contrast was +97.07 percentage points, with a paired 95% interval of +95.31 to +98.44. All four native choices were correct in every generated file.

![The edit reverses both letters in most reads of the selected class and almost no reads of the other class](binding.png)

## What the paired reads test

A generated file defines two classes with the same attribute spelling and distinct values. Its final line reads the attribute from one class, with the last letter left blank. A paired version replaces the last letter in both definitions. Every file therefore gives two class queries, each in two versions. The correct letter returns the chosen class's value. The alternative raises AttributeError. The selection cue occurs after both declarations, so their states before the intervention are identical across the two reads. The model compares only the two possible final-letter tokens. [Registered protocol](../IDENTIFIER_BINDING.md) · [Executable generator and intervention](../identifier_binding.py)

We froze Qwen2.5-14B base at revision `97e1e76335b7017d8f67c08a19d103c0504298c9`. We used 512 single-definition calibration pairs to find the 25 directions after block 10, then used that basis unchanged on 256 two-definition groups (indices 24000–24255). We projected the difference between versions at one declaration and swapped just that projected change. The baseline uses the untouched model. The positive control swaps the entire selected declaration state. [How the directions were fitted](../identifier_subspace_study/README.md)

| Intervention | Read first class | Read second class | Total reversals |
|---|---:|---:|---:|
| 25 directions at selected class | 242 / 256 | 256 / 256 | 498 / 512 |
| Same edit at the unselected class | 1 / 256 | 0 / 256 | 1 / 512 |
| Full state at selected class | 250 / 256 | 256 / 256 | 506 / 512 |

A reversal requires both versions to prefer the partner letter. Each physical declaration serves as the selected and unselected source, with the same projected edit for a given version. The result holds when the selected definition occurs earlier and is farther from the query. The verification script reconstructs the statistics from the saved margins and checks equal displacement when the query changes. [Prompts, directions, margins and execution receipt](results) · [Verification script](verify.py) · [Recomputed results](audit.json)

This tests how the model uses a known source in one family of executable Python completions. Both declarations share their attribute spelling within each file, and both change between versions when constructing the source activations. The tested effect depends on where the edit is applied and which class the prompt asks to read. The design leaves general Python binding, arbitrary identifier lengths and the exact attention pathways unresolved. Prior research has used causal interventions to study entity–attribute binding in text and other symbolic relationships. [Feng and Steinhardt on binding in context](https://arxiv.org/abs/2310.17191) · [Geiger and colleagues on interchange interventions](https://arxiv.org/abs/2106.02997) · [Registered single-relay follow-up](../IDENTIFIER_DOORWAY.md)

Run `uv run --with numpy python nested_surgery/identifier_binding_study/verify.py` from the repository root. The Studio ran the benchmark in two minutes and 26 seconds using Python 3.12, MLX 0.32.4, MLX-LM 0.32.2, NumPy 2.5.5 and Tokenizers 0.23.3. The account owner proposed testing protected activation regions and changing the research direction after the fixed-reference experiments. OpenAI Codex implemented, executed and verified this experiment and assisted with writing.

[Pubroot submission and review](https://github.com/buildngrowsv/pubroot-website/issues/436)
