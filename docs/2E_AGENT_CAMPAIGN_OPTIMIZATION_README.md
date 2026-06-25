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

- [`../configs/campaigns/molecules_small.yaml`](../configs/campaigns/molecules_small.yaml)
- [`../configs/campaigns/molecules_large.yaml`](../configs/campaigns/molecules_large.yaml)
- [`../configs/campaigns/artificial_graphs_small.yaml`](../configs/campaigns/artificial_graphs_small.yaml)
- [`../configs/campaigns/artificial_graphs_large.yaml`](../configs/campaigns/artificial_graphs_large.yaml)

## Goal

The loop is designed for iterative hyperparameter campaigns, not one-off notebook
experiments. Each campaign attempt:

1. checks the latest run state, metrics, proposal, and domain logbook context,
2. records a reason for the next proposal,
3. calls OpenAI at deterministic decision points,
4. validates the strict JSON decision and any campaign YAML patch,
5. chooses a small batch of candidate configurations,
6. writes exact patched trial configs,
7. trains/evaluates those candidates sequentially in a child process,
8. scores them by feasibility violations,
9. records artifacts and metrics under `artifact/`,
10. appends an analysis block to the domain logbook,
11. uses the results to inform the next proposal.

For molecules, the primary score is `average_num_violations` on generated molecules.
For artificial graphs, the same score is used to track graph validity and feasibility.
Lower is better.

## Campaign Domains

Campaigns are separated by domain:

```text
artifact/molecules/molecules_small_YYYYMMDD_HHMMSS_<shortid>/
artifact/molecules/molecules_large_YYYYMMDD_HHMMSS_<shortid>/
artifact/artificial_graphs/artificial_graphs_small_YYYYMMDD_HHMMSS_<shortid>/
artifact/artificial_graphs/artificial_graphs_large_YYYYMMDD_HHMMSS_<shortid>/
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
campaign_result.json strict mini-batch result summary
agent_decision.json  strict OpenAI decision for this completed run
```

Campaign-level strict JSON state is also kept beside the timestamped run
folders:

```text
artifact/<domain>/<prefix>_campaign_state.json
artifact/<domain>/<prefix>_agent_decisions.jsonl
artifact/<domain>/logs/<run>.log
```

The main CLI output is intentionally terse. Full training, sampling, warnings,
and traceback output for each candidate goes to per-trial logs:

```text
artifact/<domain>/<run>/logs/trial_001.log
artifact/<domain>/<run>/logs/trial_002.log
...
```

When a trained generator exposes metric plotting, the campaign exports loss
curves beside the trial metrics:

```text
artifact/<domain>/<run>/trials/trial_001/metrics/loss_curves.pdf
```

The latest log directory and discovered loss PDFs are shown by `status`.

The domain logbooks are:

- [`../LOGBOOK_molecules.md`](../LOGBOOK_molecules.md)
- [`../LOGBOOK_artificial_graphs.md`](../LOGBOOK_artificial_graphs.md)

## OpenAI Decision Contract

`./run_nodefield_campaign run <campaign>` starts or resumes the OpenAI-backed
parent loop. The parent loop launches an internal `run-mini-batch` child process
for deterministic execution, polls that child at `runner.poll_seconds`, and calls
OpenAI only after a mini-batch completes, fails, or needs a retryable agent
decision.

OpenAI settings are configured under `agent`:

```yaml
agent:
  model: gpt-5.3-codex
  reasoning_effort: medium
  max_output_tokens: 2000
  api_key_env: OPENAI_API_KEY
```

The response uses the OpenAI Responses API with strict structured output:

```json
{
  "decision": "no_action | update_logbook | propose_trial | stop_campaign",
  "reason": "short rationale",
  "logbook_markdown": "human-facing summary",
  "campaign_patch": "{\"agent\": {...}}"
}
```

`campaign_patch` is deliberately a JSON-encoded string. The controller decodes
it locally, validates it, and only then mutates the tracked campaign YAML.
Accepted patch paths are limited to:

- `agent.reason`
- `agent.next_attempt`
- `agent.default_trial_patch_space`
- `agent.default_trial_configs`

Patches touching `dataset`, `generation`, `runner`, `artifacts`, `logbook`, or
output roots are rejected. OpenAI quota or billing exhaustion is terminal for the
loop and writes `openai_credits_exhausted`; transient OpenAI or parse failures
write `agent_decision_failed` and are retried on the next poll.

## Proposal Modes

The controller supports two proposal styles.

Before creating a new proposal, the controller checks the latest timestamped run
for the same named campaign. This latest-result context is written to
`proposal.json` as `previous_result` and includes:

- latest run directory,
- latest status,
- latest metrics or error,
- previous proposal path,
- summary CSV path when available,
- recent logbook text.

The proposal reason combines that latest-result context with the campaign
config's `agent.reason`. This keeps the attempt reproducible even while the
range proposal itself remains config-driven.

## LLM Prompts And Polling

The shared prompt templates live in:

- [`../configs/campaigns/prompts/nodefield_campaign_proposal.md`](../configs/campaigns/prompts/nodefield_campaign_proposal.md)
- [`../configs/campaigns/prompts/nodefield_campaign_logbook.md`](../configs/campaigns/prompts/nodefield_campaign_logbook.md)

Campaign configs reference them under `agent.prompts`:

```yaml
agent:
  prompts:
    proposal: configs/campaigns/prompts/nodefield_campaign_proposal.md
    logbook: configs/campaigns/prompts/nodefield_campaign_logbook.md
```

The controller includes those prompt files, the latest metrics, the latest
campaign state, the resolved campaign config, and the recent logbook tail in the
OpenAI decision request. The resolved prompt paths are also recorded in each
run's `proposal.json`.

Polling cadence is configured per named campaign under `runner.poll_seconds`:

```yaml
runner:
  config_path: notebooks/configs/artificial_graph_hyperparameter_optimization.yaml
  poll_seconds: 1800
```

Current defaults are:

- small campaigns: `1800` seconds, 30 minutes
- large campaigns: `3600` seconds, 1 hour

`status` prints the resolved poll interval, campaign-level state, active child
PID/run, child log path, latest metrics, discovered loss PDFs, and the latest
decision/error.

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
    - model:
        fixed:
          number_of_transformer_layers: 2
    - model:
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
    - model.fixed
    - model.search_space
```

This lets the agent change model/training/search-space hyperparameters while
keeping dataset, generation, and unsafe operational paths locked. In particular,
campaign proposals should not mutate:

- `dataset`
- `generation`
- `outputs`
- artifact roots
- checkpoint roots
- arbitrary filesystem paths

Those are controlled by the campaign runner.

## Fixed Dataset And Generation Campaigns

Dataset size, graph complexity, and generation settings are fixed by the named
campaign. They are not agent-mutable trial parameters. This keeps trials within
a campaign comparable and makes "small/simple" and "large/complex" results
separate experimental conditions.

The initial named campaigns are:

```text
molecules-small             100 ZINC graphs
molecules-large            1000 ZINC graphs with larger molecular graphs
artificial-graphs-small     100 artificial graphs
artificial-graphs-large    1000 artificial graphs with larger cycle/path/star units
```

The old `molecules` and `artificial-graphs` commands remain aliases for the
small campaigns. The older artificial `simple` and `complex` aliases are also
accepted for compatibility, but new configs use `small` and `large`.

Campaign configs set fixed dataset settings directly:

```yaml
dataset:
  num_graphs: 100
  min_size: 4
  max_size: 10
  random_state: 42

generation:
  n_samples: 16
  feasibility_effort: 2
  feasibility_filter: none
```

For artificial graphs, campaign-level complexity can include:

```yaml
dataset:
  num_graphs: 1000
  cycle_length: 6
  path_length: 4
  num_rays: 3
  ray_length: 2
```

To change dataset scale or complexity, create a new named campaign config rather
than allowing the agent to modify `dataset` inside a campaign run. Use the same
pattern for generation settings such as `n_samples` and `feasibility_effort`.

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

The `list` command prints campaign names. Use those names as arguments to
`run`, `status`, and `terminate`.

Campaign `run` defaults to CPU execution so local machines with visible but
unsupported CUDA devices still work with the simple command. Use
`--device auto` or `--device cuda` only on machines where the installed PyTorch
build supports the available GPU.

Start or resume a molecule campaign loop:

```bash
./run_nodefield_campaign run molecules-small
./run_nodefield_campaign run molecules-large
```

Run one parent-loop tick and exit:

```bash
./run_nodefield_campaign run artificial-graphs-small --once
```

Dry-run one artificial graph campaign status check without launching jobs,
calling OpenAI, or mutating files:

```bash
./run_nodefield_campaign run artificial-graphs-small --dry-run
./run_nodefield_campaign run artificial-graphs-large --dry-run
```

The internal child command is available for tests and debugging, but normal use
should go through `run`:

```bash
./run_nodefield_campaign run-mini-batch artificial-graphs-small \
  --config configs/campaigns/artificial_graphs_small.yaml \
  --run-timestamp 20260625_091011 \
  --run-id debug01
```

Check status:

```bash
python run_nodefield_campaign.py status molecules
python run_nodefield_campaign.py status molecules-large
python run_nodefield_campaign.py status artificial-graphs
python run_nodefield_campaign.py status artificial-graphs-large
```

Request termination of the latest run:

```bash
python run_nodefield_campaign.py terminate molecules
```

Status reads the campaign-level `artifact/<domain>/<prefix>_campaign_state.json`
and the latest timestamped run state.

## Logbook Contract

After a completed run and OpenAI decision, the controller writes a marked block
to the domain logbook from the strict `logbook_markdown` field. Each block
records:

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

Expose broad model hyperparameter paths when iterating locally, but keep dataset,
generation, and operational paths locked. The current campaign design allows
broad model, training, loss-weight, and sampling changes without allowing
proposals to change dataset/generation conditions, redirect outputs, or overwrite
unrelated files.
