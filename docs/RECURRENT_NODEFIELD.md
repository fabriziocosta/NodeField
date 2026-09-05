# Recurrent Energy NodeField

RENF is opt-in. `ConditionalNodeFieldGenerator(node_field_mode="baseline")` retains the original energy, loss, parameter initialization and sampling path. Old saved generators without a mode use the baseline. `node_field_mode="recurrent_energy"` adds shared node-wise memory to the same Transformer and scalar potential head.

```python
from conditional_node_field_graph_generator import (
    ConditionalNodeFieldGenerator,
    RecurrentIntervention,
)

node_generator = ConditionalNodeFieldGenerator(
    node_field_mode="recurrent_energy",
    recurrent_training_steps=8,
    recurrent_detach_interval=4,
    recurrent_corruption_schedule="annealed",
)
# Fit through the existing graph generator or node-generator setup/fit workflow.
# Once fitted:
# nodes, trajectory = node_generator.predict_recurrent(
#     graph_conditioning,
#     total_steps=32,
#     intervention=RecurrentIntervention("fresh_x_noise_every_step", seed=0),
#     return_trajectory=True,
# )
```

`predict()` still returns `GeneratedNodeBatch`; module `generate()` still returns a tensor. Explicit `predict_recurrent()` / `generate_recurrent()` return `(output, trajectory)` only when capture is requested. Their ordinary outputs use the existing scaling and graph decoder. Pass a list of interventions to reset both x and h. The module additionally accepts classifier or regression guidance callbacks through the existing classifier-guidance arguments.

## Training semantics

The hidden dimension defaults to the latent dimension. Each training iteration draws a fresh corrupted clean input; only hidden state persists. Sigma and iteration indices are never network inputs. The score is the partial derivative `-∂phi/∂x` at fixed h, implemented by differentiating a distinct x branch while preserving outer gradients through recurrent history.

Denoised structural readout uses `(x + sigma² score, h_k)` and does not advance h again. Every enabled structural objective is evaluated at each supervised step, including edges, labels, locality, counts and degree consistency. Weighted losses are normalized over supervised steps; final-only supervision retains the preceding hidden computation. The loss discount must be positive.

Defaults are eight steps, zero initialization, hidden normalization, update scale 1, equal step weights, and truncated backpropagation every four updates. Set the detach interval to `None` for full backpropagation through memory. Annealed corruption decreases geometrically from `node_field_sigma` (default 0.2) to 0.02; a one-step schedule uses the maximum. Constant corruption uses the maximum throughout. The `none` schedule explicitly disables the score objective and trains structural objectives on clean inputs; it never divides by zero.

No extra parameters are instantiated in baseline mode. RENF parameters are shared across all depths. Changing inference depth does not change parameter count.

## Interventions and trajectories

Interventions run **before** the specified zero-based field evaluation. `step=4` changes the input to the fifth evaluation. `every_step=True` repeats hidden resets or shuffles. `fresh_x_noise_every_step` repeats replacement automatically. Replacement standard deviation is `noise_scale` in scaled feature coordinates, independent of Langevin noise. Intervention RNGs are local and default to seed zero; sampling RNG consumption does not change when an intervention is enabled.

Shuffling stays within each graph and only permutes valid node slots when a mask is supplied. Padding remains zero. Unconditional and conditional CFG branches have independent persistent memories and receive matched intervention randomness. Classifier/regression guidance remains separate from CFG.

`recurrent_readout(x, h, condition)` does not advance memory. It returns embeddings, latent tokens, potential, and enabled structural outputs; disabled heads are `None`.

Captured trajectories store detached CPU tensors. `x` and `h` contain the initial state and each completed update (K+1 entries); `evaluated_x`, `evaluated_h`, scores and potentials contain the K actual field inputs/outputs. This distinguishes a replacement from the sampler's update. `sigma` is zero during sampling because the sampler has no corruption schedule; replacement distributions are recorded in intervention events. Diagnostics use RMS norms, consecutive cosines and probability changes; undefined initial values are missing. Update deltas are measured after interventions, and the stored pre-intervention states allow measuring intervention jumps separately.

Capture adds readouts and tensor copies. Field evaluations, diagnostic readouts, final readouts and timings are reported separately. Trajectory capture is disabled by default. Small state changes describe **empirical stabilization**, not proof of convergence.

## Notebook defaults

The nine notebooks that construct models expose every recurrent option in an editable
`NODE_FIELD_CONFIG` dictionary next to the builder call. They default to
`node_field_mode="recurrent_energy"`. The artificial and ZINC hyperparameter-search
YAML files also declare the recurrent settings under `model.fixed`.

`build_graph_generator()` explicitly accepts and forwards all recurrent
hyperparameters and the mode. Its default remains `baseline` for existing scripts;
notebooks opt into recurrence explicitly. `None` for hidden dimension follows the
latent width, and `None` for maximum sigma follows `node_field_sigma`.

Notebook model names include the mode. New target-similarity runs start without
an automatically selected checkpoint. Sampling a previously saved generator uses
that generator's stored architecture. The ablation notebook retains its explicit
baseline and recurrent comparison matrix.

## Experiments

Open [the ablation notebook](../notebooks/recurrent_energy_nodefield_ablation.ipynb), or run:

```bash
python -m conditional_node_field_graph_generator.extensions.demo.recurrent_experiments
```

The default is the requested smoke experiment: 100 cycle/path/star graphs, cached 80/10/10 split, seed 0, ten epochs, and depths 1–32. The full matrix is prepared behind `RUN_FULL=True` in the notebook or `--full` in the command. It uses 1,000 graphs and five training seeds. Model D is an alias of C's validation-selected checkpoint, never a separate training run. Full mode includes reset/shuffle, the two-channel reset grid, no-persistent-memory control, noise regimes, matched-update curriculum comparisons and validation-calibrated anytime stopping.

Configurations live under `configs/recurrent_nodefield/`. All run directories are unique. They retain dataset/split indices, preprocessing, resolved model and decoder settings, every epoch checkpoint, selected generators, training logs, per-attempt results, trajectories, diagnostics, and regenerated figures. Failures remain in the primary metric denominator. Empty structural components are measured as zero counts, not decoder exceptions.

The primary metric requires feasible decoding and exact measured node/edge and cycle/path/star structure. Labels are compared as distributions because generated nodes are not aligned to reference slots. Parameter counts and equal field-evaluation comparisons are reported; exact parameter matching is deferred. Annealed and constant raw score losses have different target scales and should not be interpreted as comparable graph quality.

The smoke run tests numerical and workflow behavior, not significance. Full summaries aggregate independent training seeds, with paired seed-level confidence intervals. Full evidence requires a positive paired 95% interval and at least 0.05 absolute primary-metric improvement. Secondary interventions and stopping experiments are exploratory.

## Validation

```bash
pytest tests/test_recurrent_node_field.py tests/test_recurrent_experiments.py tests/test_cfg_guidance.py tests/test_sparse_supervision.py -q
pytest -q
```

The baseline fixture predates this extension and checks initialization, scores, losses, parameter gradients, generation and cached decoder heads. Additional tests cover fixed-h finite differences, shared ancestry, truncation, all structural heads, padding, inference-mode validation, interventions, capture neutrality and checkpoint reloads.
