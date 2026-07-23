# NodeField Campaign Logbook Prompt

Write a concise campaign logbook block after a run completes, fails, or is
semantically stopped during polling.

Inputs available to you:
- run directory
- proposal reason and sampled exact patches
- trial metrics
- loss PDF paths
- trial log paths
- previous result context
- scientific belief state and deterministic observation IDs

Include:
- one plain-English paragraph explaining how the run went and what conclusion to draw
- a compact Markdown table for metrics, especially `average_num_violations`, `median_num_violations`, and `feasible_rate`
- failed trial/error details when relevant, summarized in prose or in the table
- one plain-English paragraph explaining what to try next and why
- relevant observation, hypothesis, or belief IDs when making a causal claim
- artifact links only in the deterministic `Files to inspect` section appended by the controller

Do not paste long file names or absolute paths in the narrative text. Refer to
artifacts by short labels such as "the proposal", "the trial log", or "the loss
PDF"; the controller appends the actual Markdown links in `Files to inspect`.

Keep the summary short and operational. Do not repeat full logs, raw JSON, or
long sampled configurations.
