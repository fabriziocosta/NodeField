# NodeField Campaign Logbook Prompt

Write a concise campaign logbook block after a run completes or fails.

Inputs available to you:
- run directory
- proposal reason and sampled exact patches
- trial metrics
- loss PDF paths
- trial log paths
- previous result context

Include:
- what was tried
- why it was tried
- best metric values, especially `average_num_violations`, `median_num_violations`, and `feasible_rate`
- failed trial/error details when relevant
- artifact links: proposal, configs, metrics, logs, and loss PDFs
- the next proposed range attempt

Keep the summary short and operational. Do not repeat full logs.

