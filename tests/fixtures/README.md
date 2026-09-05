# Baseline numerical reference

`nodefield_baseline.pt` was captured from the original implementation at revision `1a5fe44`, before the RENF edits. It contains a tiny float32 CPU model with two examples, three node slots, one Transformer layer, no dropout, and enabled node/edge labels and locality.

Initialization seed: 1701. Loss corruption seed: 99. Generation seed: 100, with two sampling steps. Inputs, masks, labels, constructor arguments, parameter tensors, gradients, losses, scores, potentials, sampled states and final head caches are stored in the fixture. The regression test uses `torch.load(..., weights_only=True)` and checks both default and explicit baseline mode. Do not regenerate the fixture to make a regression pass.
