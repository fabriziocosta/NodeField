# NodeField Artificial Graph Campaign Logbook

This logbook tracks only the current useful small artificial campaign state. Older
artifact directories were cleaned during restarts, so stale file links have been removed.

## Useful Baselines

The best observed overall product-score region came from the early compact 1-layer/4-head
small-artificial campaign basin.

| Run | Product score | Avg violations | Embedding distance | Median violations | Feasible rate | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260630_103823 | 1.5330 | 36.6875 | 0.0418 | 35.0 | 0.0 | Best product score seen. |
| 20260630_131028 | 1.5396 | 30.5 | 0.0505 | 30.5 | 0.0 | Similar product score, better violations. |
| 20260630_134055 | 5.6071 | 21.5 | 0.2608 | 17.0 | 0.0 | Lower violations but much worse embedding distance. |
| 20260701_113018 | 2.4114 | 113.4375 | 0.0213 | 118.5 | 0.0 | 300 graphs preserved distance but collapsed violations under effort 2. |

Interpretation: do not chase raw violation count alone. The later 21.5-violation run had
poor embedding distance, and the 300-graph effort-2 restart had severe oracle-refinement
timeouts plus very high violations.

## Current Restart

Restart the small artificial campaign from the original low-product-score basin, but use
300 training graphs and effort 1 to reduce slow/high-variance oracle refinement during
32-sample scoring.

| Setting | Value |
| --- | ---: |
| Training graphs | 300 |
| Generated scoring samples | 32 |
| Feasibility effort | 1 |
| Feasibility filter | none |
| Transformer layers / heads | 1 / 4 |
| Latent / node SVD / graph SVD | 64 / 32 / 32 |
| Dropout | 0.07231 |
| Learning rate | 0.000175 |
| Batch size | 16 |
| Total steps | 307 |
| Degree weight | 0.741675 |
| Node-count weight | 0.913875 |
| Auxiliary-edge weight | 1.84485 |
| Sparse supervision mask ratio | 0.181086 |
| Sampling step size | 0.0192201 |

Next action: evaluate whether effort 1 avoids oracle timeout behavior while keeping the
embedding-distance advantage of the original low-product-score basin.

<!-- nodefield-campaign:artificial_graphs_small_20260701_165937_2c81e0:begin -->
### artificial_graphs - artificial_graphs_small_20260701_165937_2c81e0

This run completed after testing 1 candidate(s). The best candidate was trial_001 with product score 4.064370312048155, average violations 114.875, embedding distance 0.035380807939483394, median violations 118.0, and feasible rate 0.0; use the table below to compare the sampled candidates.

Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Restart the small artificial campaign from the original low-product-score basin using 300 training graphs, 32 generated scoring samples, and feasibility effort 1 to avoid slow/high-variance oracle refinement.

Next, Run one exact trial at the midpoint of the original tight range around the product-score 1.53 attempt.

#### Metrics

| Trial | Status | Product score | Avg violations | Embedding distance | Median violations | Feasible rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trial_001 | completed | 4.064370312048155 | 114.875 | 0.035380807939483394 | 118.0 | 0.0 |

#### Files to inspect

- run: [artificial_graphs_small_20260701_165937_2c81e0](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_small_20260701_165937_2c81e0/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_small_20260701_165937_2c81e0:end -->
