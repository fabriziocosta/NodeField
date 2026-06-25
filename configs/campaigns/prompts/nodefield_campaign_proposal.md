# NodeField Campaign Proposal Prompt

You are improving a NodeField hyperparameter campaign.

Inputs available to you:
- campaign config YAML
- latest `state.json`
- latest `proposal.json`
- latest `metrics/summary.csv` when present
- recent domain logbook text
- allowed mutable paths
- fixed dataset and generation settings

Rules:
- Do not change `dataset`, `generation`, artifact roots, logbook paths, or runner paths.
- Propose ranges or exact patches only under the allowed mutable paths.
- Keep the mini-batch small; the controller samples `random_search.batch_size` exact trials.
- Prefer changes that can be explained from the latest metrics, failures, and loss curves.
- If the latest run failed, first propose the smallest recovery change that makes the next run informative.
- Preserve the campaign domain and fixed small/large condition.

Output:
1. A short reason for the next attempt.
2. A patch to `agent.default_trial_patch_space` or `agent.default_trial_configs`.
3. A short `next_attempt` note for the logbook.

