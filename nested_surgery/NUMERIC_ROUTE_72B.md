# Can a transplant change which branch a 72B model follows?

Change one number in the model's internal state and ask it to read a different variable. Run an identical transplant in a mirrored program and require the answer to change in the opposite direction. A third program moves the threshold so both selector values take the same branch. Its answer should stay put.

```python
def fetch():
    mode = 2
    left = 3
    right = 7
    if mode < 4:
        return left + 1
    return right + 1
```

In this schematic, exchanging the selector from 2 to 6 changes the output from 4 to 8. Swap the two payloads in a separate prompt and the same selector exchange must change 8 to 4. Move the threshold to 7 and the exchange must preserve 4. The generated benchmark also excludes answers that appear as numeric literals anywhere in the prompt.

The **selector** is the value checked by the condition. The **payloads** are the two values the condition can choose. The intervention writes only the selector token's hidden state after block 17. Since the selector is assigned first and the transformer is causal, its state cannot depend on the prompt's later payloads or threshold. All three programs receive the same donor coordinates for any one selector value. Python execution checks every expected answer.

Use the existing seven-dimensional alignment from Qwen2.5-Coder-7B into Qwen2.5-72B. Keep both pretrained checkpoints and the fitted alignment frozen. Compare the unmodified checkpoint, the donor edit, an equal-norm edit in the archived orthogonal space, and a full hidden-vector swap from the paired prompt at the same selector site. Rank answer tokens against the whole vocabulary.

Qualify on 64 new groups, requiring at least 48 where the unmodified model is right on all six prompts. Use a disjoint panel of 256 groups. Count native-qualified groups, bidirectional switches in both mirrored contexts, and exact answer preservation in the inert context. Require at least 100 native-qualified groups, at least 80% bidirectional switches for both the donor and full-state arms, at least 95% donor preservation on inert prompts, and at least a 20-point paired joint advantage for the donor over the orthogonal edit with a positive 95% lower bound. The joint score requires all six answers to behave as predicted. Archive the qualification panel even if it falls below threshold.

The paired intervention design builds on causal abstraction. This test asks whether a previously fitted, cross-model operand interface carries a number the recipient can also use in a branch decision. [Geiger et al., Causal Abstractions of Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html)
