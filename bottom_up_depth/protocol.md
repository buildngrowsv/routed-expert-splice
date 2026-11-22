# E5 protocol: building a network from the bottom up

Registered 20 November 2026, before any full run. Code: `e5/grow.py`, reusing the E3 model, data and tokenizer.

## Question

Can a language model be built four blocks at a time, with each finished stage frozen as a read-only foundation for the next? This is the owner's bottom-up training idea, and it is also a tunnel: frozen early layers that later layers can only read. What does it cost against training the whole depth at once?

## Arms

All arms use the E3 main configuration (width 768, 12 blocks, the 8,192-entry tokenizer) and the same mixture order as the E3 dense arm.

- **Dense:** `prog-e3-dense`, all 12 blocks trained together on 400M tokens.
- **Grown, frozen (`prog-e5-grow-frozen`).**
  - Stage 1 trains a 4-block model.
  - Stage 2 adds four blocks on top. Their output projections start at zero, so the model computes the same function at the moment of growth. The bottom four blocks and the embedding are then frozen.
  - Stage 3 adds four more and freezes the bottom eight.
  - The final norm and unembedding keep training.
- **Grown, unfrozen (`prog-e5-grow-unfrozen`):** the same stages with every weight trainable throughout, as in progressive stacking.

Each stage gets one third of the dense arm's tokens. Each stage has its own warmup and cosine schedule with peak learning rate 6e-4.

## Measurements

- **Loss:**
  - mean validation loss over English, Python and Welsh, plus each domain separately;
  - Icelandic loss as an unseen-domain check.
- **Training compute:** frozen blocks cost a forward pass only, so a grown model reaches the end with fewer FLOPs than dense. Results are reported against both tokens and FLOPs.
- **Equal-FLOP comparison:** the grown model's final loss against the dense loss curve interpolated at the same FLOPs. The equal-token comparison against dense's final loss is also reported. The dense curve at an intermediate point has not finished its learning-rate decay, which favours growth. The equal-token number is the conservative one.

## Success

- **`growth_matches`:** the frozen-growth model is within 3% of dense loss at equal FLOPs.
- **`growth_costs`:** otherwise.

The unfrozen arm separates the cost of freezing from the cost of growing.

## Pictures

The atlas timelapse spans all three stages, with upper blocks shown as pass-throughs before they exist. Comparing against the end of stage 1 shows the frozen foundation staying dark, while the unfrozen arm keeps rewriting it.

## Prior work

- Layer-wise greedy pretraining (Bengio et al., 2007).
- Progressive stacking (Gong et al., 2019).
- Function-preserving growth (Net2Net, Chen et al., 2016).
- LLaMA Pro block expansion (Wu et al., 2024).

## Budget

About 2 hours per arm on the Studio, after `prog-e3-dense`.
