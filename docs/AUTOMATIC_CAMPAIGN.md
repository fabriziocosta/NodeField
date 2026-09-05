# Automatic NodeField campaigns

An automatic campaign is a controlled loop that trains a sequence of NodeField
experiments, records what happened, and selects the next experiment from the
remaining scientific candidates. The controller runs experiments sequentially
and asks the language model for decisions only at configured poll or run
boundaries.

The campaign is not a notebook workflow. Campaign YAML fixes the dataset,
generation policy, artifact locations, budgets, and parameter allowlist. The
mutable scientific memory is stored separately in:

```text
artifact/<domain>/<prefix>_state.yaml
```

## What happens automatically

```text
campaign YAML
      |
      v
initialize or migrate state.yaml
      |
      v
validate candidates and budget
      |
      v
launch one copied trial config
      |
      v
write epoch telemetry and detect observations
      |
      v
record immutable experiment + observations
      |
      v
apply typed belief-state operations
      |
      v
rank and launch the next valid candidate
```

The model objective is unchanged by the controller. During training, the
scientific metrics callback writes compact epoch records and deterministic
observations. Full curves remain in trial artifacts. Typical observations are:

- validation plateaus and diminishing returns;
- train/validation generalisation gaps;
- non-finite metrics;
- unstable gradients;
- anomalous epoch runtime;
- checkpoint and termination provenance.

Non-finite metrics and severe instability can produce a policy-driven stop
signal while preserving existing early-stopping and best-checkpoint restoration
behavior.

## Scientific state

`state.yaml` contains the current belief state rather than a mutable logbook. It
includes:

- immutable experiments and observations;
- hypotheses, beliefs, and open questions;
- candidate experiments and their evidence;
- controlled relations such as `supports`, `contradicts`, `tests`, and
  `replicates`;
- controller state, trigger rules, active run, and remaining budget.

Completed experiments and observations cannot be rewritten. Updates to mutable
entities use typed operations and optimistic previous values. Candidate
experiments must specify fixed parameters, varied parameters, seeds, expected
discriminating outcomes, estimated cost, and risk mitigation.

The controller validates every candidate against the campaign parameter
allowlist, remaining and per-experiment budget, maximum run duration,
replication requirements, and concurrency limits. Valid candidates are ranked
deterministically by scientific value per estimated cost, with candidate ID as
the stable tie-breaker. Version 1 runs one active experiment at a time.

## Campaign configuration

Built-in campaigns are defined in [`configs/campaigns`](../configs/campaigns):

- [`molecules_small.yaml`](../configs/campaigns/molecules_small.yaml)
- [`molecules_large.yaml`](../configs/campaigns/molecules_large.yaml)
- [`artificial_graphs_small.yaml`](../configs/campaigns/artificial_graphs_small.yaml)
- [`artificial_graphs_large.yaml`](../configs/campaigns/artificial_graphs_large.yaml)

A scientific campaign normally enables:

```yaml
scientific_loop:
  primary_metric: average_num_violations
  budgets:
    maximum_single_experiment_gpu_hours: 20
    maximum_parallel_runs: 1

agent:
  stateful_loop: true
```

The campaign YAML is the reproducible execution and policy definition.
Candidate patches are applied to copied trial configurations; the base campaign
configuration is not changed by an experiment.

## Running a campaign

From the repository root:

```bash
./run_nodefield_campaign list
./run_nodefield_campaign run molecules-small
```

Useful variants:

```bash
# Execute one controller tick and return.
./run_nodefield_campaign run molecules-small --once

# Inspect status without launching a job or calling the model.
./run_nodefield_campaign run molecules-small --dry-run

# Show campaign, belief-state, candidate, observation, and artifact status.
./run_nodefield_campaign status molecules-small

# Request termination of the active campaign child.
./run_nodefield_campaign terminate molecules-small

# Clean up stale state and start a fresh run.
./run_nodefield_campaign force-restart molecules-small
```

`run-mini-batch` is the deterministic child command used by the parent loop. It
is normally not called manually, but it is useful for reproducing a specific
resolved trial with its recorded run timestamp and candidate ID.

## Artifacts and inspection

Each campaign has a domain-specific artifact directory. A typical trial contains:

```text
artifact/<domain>/<campaign-run>/
├── configs/       resolved campaign and trial YAML
├── trials/        one directory per experiment
├── metrics/       summary metrics and epoch telemetry
├── logs/          child and trial logs
├── samples/       generated samples and previews
└── state.json     execution status for this run
```

Inside a trial, inspect:

```text
metrics/epoch_telemetry.jsonl
metrics/observations.jsonl
metrics/loss_curves.csv
metrics/loss_curves.pdf
```

The YAML belief state stores summaries and evidence references, not full metric
curves. Human-readable logbooks are generated views of campaign changes and
should not be treated as the primary scientific memory.

## Reading `status`

`status` reports the current controller state, including the active process and
polling interval, campaign and scientific state locations and schema version,
belief count, active hypotheses, open questions, pending and selected
candidates, recorded observations, latest metrics, decisions, errors, logs, and
checkpoint artifacts.

Use the state YAML and trial JSONL files when the compact status output is not
enough to audit a decision.

## Restart and failure behavior

The loop resumes from filesystem state. If a child exits without writing a
final state, the parent records the failure and asks for a new decision. A
failed or malformed scientific decision is recorded as `agent_decision_failed`
and can be retried with `run ... --once`.

OpenAI quota or billing exhaustion is treated as terminal rather than retried
indefinitely. User termination preserves partial trial evidence.
`force-restart` records cleanup of stale process state before launching a new
campaign run.

For detailed configuration and legacy migration behavior, see the
[agent campaign optimization reference](2E_AGENT_CAMPAIGN_OPTIMIZATION_README.md).

To inspect the newest campaign without supplying a campaign name, open the
[automatic campaign state dashboard](../notebooks/campaigns/monitor.ipynb).
