# Agent Campaign Optimization Loop

This document describes the NodeField campaign loop used to optimize graph-generation
hyperparameters across the two supported domains:

- molecules
- artificial graphs

The campaign implementation lives in:

- [`../conditional_node_field_graph_generator/nodefield_campaign.py`](../conditional_node_field_graph_generator/nodefield_campaign.py)
- [`../conditional_node_field_graph_generator/campaign_search.py`](../conditional_node_field_graph_generator/campaign_search.py)
- [`../run_nodefield_campaign.py`](../run_nodefield_campaign.py)

The default campaign configs live in:

- [`../configs/campaigns/molecules.yaml`](../configs/campaigns/molecules.yaml)
- [`../configs/campaigns/artificial_graphs.yaml`](../configs/campaigns/artificial_graphs.yaml)

## Goal

The loop is designed for iterative hyperparameter campaigns, not one-off notebook
experiments. Each campaign attempt:

1. chooses a small batch of candidate configurations,
2. trains/evaluates those candidates sequentially,
3. scores them by feasibility violations,
4. records artifacts and metrics under `artifact/`,
5. appends an analysis block to the domain logbook,
6. uses the results to inform the next proposal.

For molecules, the primary score is `average_num_violations` on generated molecules.
For artificial graphs, the same score is used to track graph validity and feasibility.
Lower is better.

## Campaign Domains

Campaigns are separated by domain:

```text
artifact/molecules/molecules_YYYYMMDD_HHMMSS_<shortid>/
artifact/artificial_graphs/artificial_graphs_YYYYMMDD_HHMMSS_<shortid>/
```

Each run contains:

```text
configs/      resolved campaign and trial configs
trials/       one folder per candidate trial
logs/         stdout/stderr or future agent logs
metrics/      summary JSON/CSV files
samples/      generated graph or molecule previews
state.json    machine-readable campaign status
proposal.json proposed ranges/configs and sampled exact patches
```

The domain logbooks are:

- [`../LOGBOOK_molecules.md`](../LOGBOOK_molecules.md)
- [`../LOGBOOK_artificial_graphs.md`](../LOGBOOK_artificial_graphs.md)

## Proposal Modes

The controller supports two proposal styles.

### `range_search`

In `range_search`, the agent proposes ranges. The controller samples a small number
of exact trial patches from those ranges.

This is the default because it separates strategic exploration from concrete
execution:

- the agent says what region to explore,
- the controller deterministically samples exact candidates,
- training sees only exact resolved configs.

Example:

```yaml
agent:
  proposal_mode: range_search
  default_trial_patch_space:
    model:
      fixed:
        number_of_transformer_layers:
          type: int
          low: 1
          high: 4
        latent_embedding_dimension:
          type: choice
          values: [32, 64, 128]
```

The mini-batch size is controlled by:

```yaml
random_search:
  batch_size: 3
  random_state: 101
```

### `exact_configs`

In `exact_configs`, the agent supplies complete candidate patches directly.

This is useful for ablations, follow-up checks, or retrying known-good regions
without random sampling.

Example:

```yaml
agent:
  proposal_mode: exact_configs
  default_trial_configs:
    - dataset:
        num_graphs: 500
      model:
        fixed:
          number_of_transformer_layers: 2
    - dataset:
        num_graphs: 750
      model:
        fixed:
          number_of_transformer_layers: 3
```

Exact configs are still validated against the campaign allowlist.

## Mutable Groups

Campaign configs expose hyperparameters through mutable groups. These are
allowlist shortcuts that determine which parts of the workflow an agent may
change.

Supported groups:

```yaml
agent:
  mutable_groups:
    - dataset
    - generation
    - architecture
    - training
    - loss_weights
    - sampling
```

The default campaigns also allow broad hyperparameter edits through explicit
paths:

```yaml
agent:
  allowed_paths:
    - dataset
    - generation
    - model.fixed
    - model.search_space
```

This lets the agent change all model/training/search-space hyperparameters while
keeping unsafe operational paths locked. In particular, campaign proposals should
not mutate:

- `outputs`
- artifact roots
- checkpoint roots
- arbitrary filesystem paths

Those are controlled by the campaign runner.

## Dataset And Complexity Control

Campaign configs may set fixed dataset settings directly:

```yaml
dataset:
  num_graphs: 500
  min_size: 4
  max_size: 10
  random_state: 42
```

For artificial graphs, campaign-level complexity can include:

```yaml
dataset:
  num_graphs: 300
  cycle_length: 6
  path_length: 4
  num_rays: 3
  ray_length: 2
```

The agent can also explore these as ranges:

```yaml
agent:
  default_trial_patch_space:
    dataset:
      num_graphs:
        type: int
        low: 150
        high: 600
      cycle_length:
        type: int
        low: 4
        high: 9
```

This supports curriculum-style campaigns: start with smaller/easier graphs,
tune stability and feasibility, then expand graph size or structural complexity.

## Architecture Search

The campaign loop can explore architectural changes when the `architecture`
group or `model.fixed` path is enabled.

Typical architecture knobs include:

- `latent_embedding_dimension`
- `node_embedding_svd_dimension`
- `graph_embedding_svd_dimension`
- `number_of_transformer_layers`
- `transformer_attention_head_count`
- `transformer_dropout`
- `locality_horizon`
- `locality_sample_fraction`

Example:

```yaml
agent:
  default_trial_patch_space:
    model:
      fixed:
        latent_embedding_dimension:
          type: choice
          values: [32, 64, 128]
        number_of_transformer_layers:
          type: int
          low: 1
          high: 4
        transformer_attention_head_count:
          type: choice
          values: [2, 4, 8]
```

Architecture search should be used deliberately because it changes model
capacity and makes trials less directly comparable than pure loss-weight or
sampling sweeps.

## Trial Resolution

Each trial starts from the base workflow YAML:

- molecules:
  [`../notebooks/configs/zinc_molecule_hyperparameter_optimization.yaml`](../notebooks/configs/zinc_molecule_hyperparameter_optimization.yaml)
- artificial graphs:
  [`../notebooks/configs/artificial_graph_hyperparameter_optimization.yaml`](../notebooks/configs/artificial_graph_hyperparameter_optimization.yaml)

The controller then applies:

1. fixed campaign-level overrides such as `dataset`, `generation`, or `model`,
2. the sampled exact trial patch,
3. campaign-managed output paths under `artifact/`.

For entries under `model.search_space`, exact sampled values are converted into
one-point ranges:

```yaml
sampling_step_size:
  type: real
  low: 0.047
  high: 0.047
```

This keeps compatibility with the existing random-search notebook helpers while
making each campaign candidate reproducible.

## CLI

List campaigns:

```bash
python run_nodefield_campaign.py list
```

Run one molecule mini-batch:

```bash
python run_nodefield_campaign.py molecules --once
```

Dry-run one artificial graph mini-batch:

```bash
python run_nodefield_campaign.py artificial-graphs --once --dry-run
```

Check status:

```bash
python run_nodefield_campaign.py status molecules
python run_nodefield_campaign.py status artificial-graphs
```

Request termination of the latest run:

```bash
python run_nodefield_campaign.py terminate molecules
```

Status reads only the latest `artifact/<campaign>/.../state.json`.

## Logbook Contract

After a completed run, the controller writes a marked block to the domain
logbook. Each block records:

- what was tried,
- proposal mode,
- mutable groups,
- agent reasoning,
- link to `proposal.json`,
- latest metrics,
- artifact path,
- next proposed attempt.

The markers allow the same run block to be upserted if a run is retried or
updated:

```text
<!-- nodefield-campaign:<run-id>:begin -->
...
<!-- nodefield-campaign:<run-id>:end -->
```

## Practical Guidance

Use `range_search` when the next step is exploratory. Keep `batch_size` small
(usually 2 or 3) so the campaign can react after each mini-batch.

Use `exact_configs` when the next step is confirmatory, such as rechecking a
promising configuration or running a controlled ablation.

Expose broad hyperparameter paths when iterating locally, but keep operational
paths locked. The current campaign design allows broad model and dataset changes
without allowing proposals to redirect outputs or overwrite unrelated files.
