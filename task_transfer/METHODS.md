# Executable repair and controlled donor withdrawal

## Problem generation and screening

Three registered prefix-completion tasks use code with four legal repairs. In the bit task each operator is applied to integer operands. The list task aggregates a filtered, shifted sequence. The SQLite task evaluates aggregates against an in-memory table. Every candidate is executed on twelve concealed inputs. The public generators deterministically regenerate input values, expected outputs, code and visible examples. The expected values come from reference functions implemented separately from the candidate programs.

The first follow-up embeds a missing Python bit operator between the Qwen fill-in-the-middle prefix, suffix and middle markers. Source, three worked assertions and a short operation description are visible. The second follow-up repairs one missing token from a Python function. Its candidate alphabet is a closing parenthesis, closing square bracket, closing brace and colon. A function call, indexing expression, dictionary construction or conditional supplies the missing site, with three variants per token, varied constants, field names and a visible example. Each program takes a list and returns a number. The twelve concealed tests use new lists. Candidate programs that fail to parse fail all tests. A solve passes all twelve. The parser baseline takes the first patch accepted by Python's compiler without consulting outputs.

Each screen is balanced by the required patch. All models see identical prompts and a shared Qwen tokenizer. We restrict decoding to the four legal token IDs, take the largest raw model logit and score that patch. The logits are taken at the true end of each right-padded prompt. This task tests constrained single-token repair. Evaluation models never see IDs or concealed test cases. The delimiter training and final evaluation use the native infill representation. A separate standalone comparison renders exactly the same final functions and examples as plain text with a visible hole.

Task discovery uses 128 instances per family. The highest-scoring original candidate must lead the larger model by five percentage points to advance. Confirmation uses 256 fresh instances and a two-sided exact paired McNemar test on donor-only versus host-only solves. Each follow-up independently reuses that rule on new data after recording the preceding screen. The 384-problem final panel was withheld during selection. The qualification reports describe this adaptive search, with the final panel providing a further independent comparison. Program forms recur across splits. Complete prompts and completed executable sources are checked for duplicates across all 2,560 delimiter items.

## Graft and training

The base host is Qwen2.5-1.5B. Its feed-forward network in one-based block 27 is replaced with the same final feed-forward network from Qwen2.5-Coder-0.5B used previously. That donor contains 13,074,432 parameters. The input and output affine maps were fitted in the prior CPython study and reused, frozen, for every task run. Their SHA-256 hash is saved. Corresponding controls retain the native feed-forward network, set its output to zero, use a direct fitted affine map, use the neuron-shuffled coding module with its own fitted output map, or use an intact general-purpose Qwen2.5-0.5B module and its maps. These are mandatory branches per token. No input router is trained. [Prior methods and pinned model revisions](../penultimate/METHODS.md)

Only the complete following block, block 28, can learn. Its 46,797,824 trainable parameters include both normalizations, attention and feed-forward weights. All six arms start from its original weights. Training now minimizes cross-entropy over the four task patches. Graft inputs across the full sequence are cached on the Studio. During scoring the actual module is installed and the ordinary host forward pass supplies the logits. The shared budget is one epoch over 1,536 problems, batch size 16, 96 Adam updates at 0.00002, bias correction enabled, FP32 master weights, BF16 forward weights and order seeds 101, 211 and 307. Each seed uses the same example order across arms. Validation loss on 256 separate problems selects among steps 0, 8, 32 and 96, retaining the earlier checkpoint on ties. Held-out outcomes are reported for all selected models. Tests and training use the same program generator and differ in source and values.

## Counting transfer

A new solve is a problem missed by the untouched host and solved by the trained host carrying the coding graft. For each seed, we keep the trained block fixed, zero the graft's output at block 27, and score the same problem again. This removes the donor and its fitted output contribution while preserving the host attention and residual. We also swap the coding, shuffled and general-purpose modules under fixed adapted host blocks. The stricter transfer count requires the standalone donor to succeed, the original host, frozen graft and task-trained native host all to fail, and donor withdrawal to erase the trained solve. The released ledger records each repair character by arm and seed. The paired solve-rate interaction compares the on-minus-off solve rate after and before task training. Intervals use 5,000 bootstrap draws over paired problem IDs and order seeds. They characterize this generator and these shared pretrained checkpoints.

## Reproduction

Use the root requirements on Apple Silicon. From a fresh checkout, reproduce the frozen CPython maps using the root preparation and penultimate feature steps. The original public window manifests ship with the repository. All later commands run from its root.

```sh
.venv/bin/python prepare.py
.venv/bin/python penultimate/features.py
.venv/bin/python task_transfer/screen.py
.venv/bin/python task_transfer/qualify_infill.py
.venv/bin/python task_transfer/qualify_delimiters.py
.venv/bin/python task_transfer/surgery.py
.venv/bin/python task_transfer/test_models.py
uv run --with numpy python task_transfer/analyze.py
uv run --with matplotlib --with numpy python task_transfer/figure.py
```

The last two commands work from the released scores on CPU. The raw logits, per-item splits, repaired programs, per-candidate tests and bootstrap summaries can be inspected without loading the models.
