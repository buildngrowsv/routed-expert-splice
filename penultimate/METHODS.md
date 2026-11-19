# Moving the coding expert one block upstream

## The operation

The Qwen2.5-1.5B base language model contains 28 transformer blocks. Each block mixes information across earlier token positions using attention, then transforms each token separately with a feed-forward network. At block 27 we keep the attention and residual connection and replace only the feed-forward network. The donor is the final feed-forward network from Qwen2.5-Coder-0.5B, with 13,074,432 pretrained weights. The larger model sends its current state through the donor for every token. Block 28 receives the resulting sequence and is the only part allowed to learn by gradient descent.

The host represents each token with 1,536 numbers. The donor expects 896. We fit an affine input map against paired host and donor activations, then fit an affine output map to reconstruct the removed host network's output. Each regression uses the original training set, centering and ridge regularization with coefficient 0.02 times the mean Gram-matrix diagonal. The graft, both maps, original attention at block 27, embeddings and output head are fixed during host adaptation. Weights and maps are shared across seed repeats. Compare the earlier last-block transplant on its own test panel. [Earlier experiment](../paper.md)

## Controls and supervision

| Block 27 condition | What enters the next host block |
|---|---|
| Native | The unchanged residual and host feed-forward output |
| Removed | The residual alone, with zero feed-forward output |
| Direct affine | The residual plus one fitted linear map with bias from host input to host feed-forward output |
| Shuffled donor | The residual plus a coding expert with its up-projection rows permuted relative to its gate and down-projection, using the original permutation seed 9917 and its own fitted output map |
| General donor | The residual plus the final feed-forward network of Qwen2.5-0.5B, with the same dimensions and fitting budget as the coding donor |
| Coding donor | The residual plus the aligned, intact coding expert |

Each condition starts from the same original block 28. All 46,797,824 parameters in its attention, feed-forward network and two normalizations can move. The target is the untouched model's final residual vector. For each scored position we minimize mean squared difference divided by the teacher vector's mean square plus 0.01. The native condition consequently starts with zero reconstruction error. This objective asks whether the host can recover its original behavior. A task-specific objective would ask a different question.

We use the same 1,536 training windows and 256 validation windows as the earlier experiment. Seeds 101, 211 and 307 change only minibatch order, paired across conditions. Each fit uses one epoch of 96 batches of 16 windows, Adam with bias correction, a constant learning rate of 0.00002, default betas 0.9 and 0.999, and epsilon 1e-8. The optimizer maintains FP32 master weights and BF16 forward weights. The full 64-token sequence enters the last attention layer. Its position-wise feed-forward computation during training is needed only at the four scored positions. Updates 0, 8, 32 and 96 are candidates. The lowest pooled validation next-token loss selects the checkpoint, with ties going to the earliest update.

## Held-out evidence

We sample 512 new code windows and 512 documentation windows from the earlier CPython held-out files, with no repeated window offsets or token hashes across panels. Training, validation and test use disjoint files. We score the next token at zero-based input positions 31, 39, 47 and 55. This gives 2,048 predictions per domain in each condition. Seeds repeat those predictions on the same windows. The split measures generalization across files within CPython. It cannot establish whether public CPython text appeared in Qwen pretraining.

Both validation selection and every published test score use the ordinary end-to-end host forward on batches of eight windows with the actual replacement installed. The cached final-block training path gave slightly different BF16 predictions in a diagnostic, so we used full-model scoring for both decisions and outcomes. The diagnostic covers 32 scored positions per condition and records mean and maximum absolute loss differences and top-one agreement. [Full-model diagnostic](results/full-model-check.json)

For the removal test we take the final block trained with the coding donor, remove the donor, leave the learned block untouched and rescore the same windows. We compare that penalty with removing the same donor before training. For the crossover, the coder-trained and shuffle-trained host blocks each see both donors. All differences use paired predictions. Intervals are the 2.5th and 97.5th percentiles of 5,000 bootstrap draws, resampling whole files within a domain and paired minibatch-order seeds across arms. All runs share one model initialization. With only three order seeds, the intervals primarily characterize variation across these source files. The primary decision rule and secondary diagnostics were committed before the new test was scored. [Registered plan](PLAN.md)

## Reproduction

From a fresh checkout, initialize the environment as in the repository README, then run the following from its root. The existing published manifests identify the earlier panels. Pinned revisions for the host, coding donor and corpus are in prepare.py. The same-sized general donor's revision is in compare_donors.py. Cached training states and checkpoints stay on the research machine. The public data include per-position loss, correctness, domain, file, window manifest, validation selections and weight-change checks.

```sh
.venv/bin/python prepare.py
.venv/bin/python penultimate/features.py
.venv/bin/python penultimate/train.py
.venv/bin/python penultimate/evaluate.py
.venv/bin/python penultimate/analyze.py
uv run --with numpy python penultimate/verify.py
uv run --with matplotlib --with numpy python penultimate/figure.py
```
