# NodeField Artificial Graph Campaign Logbook

Artificial graph campaign runs append marked blocks here with the attempted proposal, sampled exact configs, metrics, artifact links, and the next proposed range attempt.

Current logbook convention: each new run should have one conclusion paragraph, a compact metrics table, one next-attempt paragraph, and a `Files to inspect` section with the actual links to configs, logs, metrics, and loss PDFs. Long paths should stay out of the narrative text.

<!-- nodefield-campaign:artificial_graphs_large_20260626_173806_9fb463:begin -->
### artificial_graphs - artificial_graphs_large_20260626_173806_9fb463

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Explore validity on the fixed large 1000-graph artificial campaign.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/proposal.json`
- Metrics: average_num_violations=209.125, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=208.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260626_173806_9fb463](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463)
- Next attempt: Compare feasibility violations, then tighten architecture, training, loss-weight, and sampling ranges.

#### Files to inspect

- run: [artificial_graphs_large_20260626_173806_9fb463](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260626_173806_9fb463/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260626_173806_9fb463:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260627_043826_b6b154:begin -->
### artificial_graphs - artificial_graphs_large_20260627_043826_b6b154

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Observed zero-feasibility output with high violation counts in the first completed trial; shift search toward feasibility-oriented weighting and more conservative optimization/sampling.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/proposal.json`
- Metrics: average_num_violations=65.25, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=57.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260627_043826_b6b154](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154)
- Next attempt: Run one feasibility-focused candidate with stronger structural penalties, smaller sampling step size, and gentler training hyperparameters; compare violation metrics before any broader expansion.

#### Files to inspect

- run: [artificial_graphs_large_20260627_043826_b6b154](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_043826_b6b154/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260627_043826_b6b154:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260627_213846_764173:begin -->
### artificial_graphs - artificial_graphs_large_20260627_213846_764173

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Second run reduced violations sharply but still produced zero feasible graphs; prioritize crossing the feasibility boundary with stronger structure-oriented loss weights, smaller sampling steps, lower masking, and gentler optimization.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/proposal.json`
- Metrics: average_num_violations=121.625, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=117.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260627_213846_764173](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173)
- Next attempt: Run one tightly constrained feasibility-first candidate (batch_size=1 campaign) with higher structural lambdas, smaller sampling step size, lower sparse supervision masking, and lower LR/dropout; check if feasible_rate moves above 0.

#### Files to inspect

- run: [artificial_graphs_large_20260627_213846_764173](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_213846_764173/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260627_213846_764173:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260627_233905_416711:begin -->
### artificial_graphs - artificial_graphs_large_20260627_233905_416711

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest run regressed in violation counts (avg 121.625, median 117.5) while feasible_rate stayed 0.0; with batch_size=1, narrow the search around stronger structural constraints and more conservative optimization/sampling to maximize the chance of crossing into non-zero feasibility.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/proposal.json`
- Metrics: average_num_violations=223.4375, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=142.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260627_233905_416711](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711)
- Next attempt: Run one tightly narrowed feasibility-first candidate: enforce higher structural lambdas (including edge-count/consistency), smaller sampling step size, lower mask ratio, lower LR/dropout, and slightly longer total_steps; objective is first non-zero feasible_rate with improved violation metrics.

#### Files to inspect

- run: [artificial_graphs_large_20260627_233905_416711](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260627_233905_416711/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260627_233905_416711:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_013922_a05d34:begin -->
### artificial_graphs - artificial_graphs_large_20260628_013922_a05d34

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest mini-batch completed but regressed strongly (average_num_violations=223.4375, median_num_violations=142.0) while feasible_rate remained 0.0; with single-trial batches, tighten ranges to enforce stronger structural constraints and gentler optimization/sampling for a feasibility-first attempt.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/proposal.json`
- Metrics: average_num_violations=229.5625, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=166.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_013922_a05d34](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34)
- Next attempt: Run one tightly constrained feasibility-first candidate with high edge/count/consistency-related lambdas, smaller sampling step size, lower sparse supervision masking, lower LR/dropout, and longer total_steps to target first non-zero feasible_rate.

#### Files to inspect

- run: [artificial_graphs_large_20260628_013922_a05d34](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_013922_a05d34/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_013922_a05d34:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_033936_8bcfdf:begin -->
### artificial_graphs - artificial_graphs_large_20260628_033936_8bcfdf

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest completed trial regressed again (average_num_violations=229.5625, median_num_violations=166.5) with feasible_rate still 0.0; with one sampled trial per batch, tighten to a strict feasibility-first region so the sampled candidate consistently uses stronger structural constraints and gentler training/sampling.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/proposal.json`
- Metrics: average_num_violations=178.4375, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=146.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_033936_8bcfdf](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf)
- Next attempt: Run one strict feasibility-first candidate by constraining to high structural lambdas, smaller sampling step size, near-zero sparse masking, slightly lower LR/dropout, and longer total_steps to maximize chance of first non-zero feasible_rate.

#### Files to inspect

- run: [artificial_graphs_large_20260628_033936_8bcfdf](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_033936_8bcfdf/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_033936_8bcfdf:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_053951_a7340e:begin -->
### artificial_graphs - artificial_graphs_large_20260628_053951_a7340e

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest strict feasibility-first trial improved violation counts (average_num_violations=178.4375, median_num_violations=146.5) but feasible_rate remains 0.0; continue in the same direction with an even tighter high-constraint region to maximize chance of first feasible samples.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/proposal.json`
- Metrics: average_num_violations=157.8125, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=148.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_053951_a7340e](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e)
- Next attempt: Run one more ultra-constrained feasibility-first candidate: very high structural lambdas (including higher node_count), smaller sampling step size, near-zero sparse masking, slightly lower LR/dropout, and longer total_steps.

#### Files to inspect

- run: [artificial_graphs_large_20260628_053951_a7340e](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_053951_a7340e:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_053951_a7340e:agent:begin -->
- Completed 1 trial from range_search under ultra-constrained feasibility-first settings.
- Best metrics: average_num_violations=157.8125, median_num_violations=148.0, feasible_rate=0.0.
- Direction remains promising on average violations but no feasible samples yet; next step is a tighter high-constraint region to force consistently strong structural weighting.

#### Files to inspect

- run: [artificial_graphs_large_20260628_053951_a7340e](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_053951_a7340e/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_053951_a7340e:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_074007_fe5831:begin -->
### artificial_graphs - artificial_graphs_large_20260628_074007_fe5831

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest run further reduced average_num_violations to 157.8125 (from 178.4375), but feasible_rate remains 0.0 and median_num_violations is flat/slightly worse (148.0 vs 146.5). With batch_size=1, tighten ranges so sampled trials are consistently feasibility-first: higher structural lambdas, smaller sampling step size, near-zero masking, slightly lower LR/dropout, and longer training.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/proposal.json`
- Metrics: average_num_violations=316.375, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=290.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_074007_fe5831](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831)
- Next attempt: Run one tighter feasibility-first candidate with near-top structural lambdas (including higher node_count), smaller sampling_step_size, near-zero sparse masking, lower LR/dropout, and increased total_steps to target first non-zero feasible_rate.

#### Files to inspect

- run: [artificial_graphs_large_20260628_074007_fe5831](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_074007_fe5831:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_074007_fe5831:agent:begin -->
- Tried 1 `range_search` candidate from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/proposal.json) in run [artificial_graphs_large_20260628_074007_fe5831](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831).
- Why: continue feasibility-first tightening after prior improvement, but this sampled point underperformed.
- Best/only metrics from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/metrics/summary.csv): **average_num_violations=316.375**, **median_num_violations=290.0**, **feasible_rate=0.0** (trial details in [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/trial_results.csv)).
- Trial artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/logs/trial_001.log), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: low-variance rescue range with all structural lambdas (especially node_count) clamped near top values, smaller sampling_step_size, near-zero sparse masking, slightly lower LR/dropout, and longer total_steps.

#### Files to inspect

- run: [artificial_graphs_large_20260628_074007_fe5831](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_074007_fe5831/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_074007_fe5831:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_094028_b96817:begin -->
### artificial_graphs - artificial_graphs_large_20260628_094028_b96817

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest single-trial result regressed sharply (avg violations 316.375, median 290.0, feasible_rate 0.0) versus prior best, indicating batch_size=1 sensitivity to still-wide feasibility ranges; constrain to consistently high structural pressure for an informative recovery run.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/proposal.json`
- Metrics: average_num_violations=1543.0, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=1231.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_094028_b96817](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817)
- Next attempt: Run one recovery candidate with top-clamped structural lambdas (including high node_count), smaller sampling step, near-zero sparse masking, slightly lower LR/dropout, and longer training to re-approach prior low-violation regime and probe first feasible samples.

#### Files to inspect

- run: [artificial_graphs_large_20260628_094028_b96817](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_094028_b96817:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_094028_b96817:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_094028_b96817](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/proposal.json).
- Why: low-variance recovery after prior regression, emphasizing high structural penalties.
- Best/only metrics from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/metrics/summary.csv): **average_num_violations=1543.0**, **median_num_violations=1231.0**, **feasible_rate=0.0** (trial details in [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/trial_results.csv)).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/logs/trial_001.log), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: ultra-tight rescue range with node/edge structural lambdas clamped very near 5.0 (especially node_count), smaller sampling_step_size, near-zero sparse masking, slightly lower LR/dropout, and longer total_steps.

#### Files to inspect

- run: [artificial_graphs_large_20260628_094028_b96817](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_094028_b96817/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_094028_b96817:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_104046_d05d04:begin -->
### artificial_graphs - artificial_graphs_large_20260628_104046_d05d04

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest completed mini-batch regressed severely (average_num_violations 1543.0, median 1231.0, feasible_rate 0.0). With random_search.batch_size=1, remaining range variance is too high; enforce an ultra-tight feasibility-focused band near prior best structural settings to make the next single trial maximally informative.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/proposal.json`
- Metrics: average_num_violations=604.3333333333334, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=557.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_104046_d05d04](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04)
- Next attempt: Run one ultra-tight rescue candidate: clamp structural lambdas (especially node_count) to ~5.0, reduce sampling_step_size, keep sparse masking near zero, slightly lower LR/dropout, and extend training steps.

#### Files to inspect

- run: [artificial_graphs_large_20260628_104046_d05d04](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_104046_d05d04:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_104046_d05d04:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_104046_d05d04](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/proposal.json).
- Why: feasibility-focused rescue after severe regression, with high structural penalties and reduced sampling step.
- Best/only metrics from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/metrics/summary.csv): **average_num_violations=604.3333333333334**, **median_num_violations=557.0**, **feasible_rate=0.0** (details in [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/trial_results.csv)).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/logs/trial_001.log), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: exploit this improved basin with even tighter lambda clamps near 5.0, smaller sampling step size, near-zero sparse masking, and longer total_steps to push toward first feasible samples.

#### Files to inspect

- run: [artificial_graphs_large_20260628_104046_d05d04](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_104046_d05d04/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_104046_d05d04:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_124104_6e7e95:begin -->
### artificial_graphs - artificial_graphs_large_20260628_124104_6e7e95

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest rescue trial significantly reduced violations (avg 604.33, median 557.0) versus the prior run (avg 1543.0, median 1231.0), but feasible_rate remains 0.0. Next, run a tighter exploit around this improved region with stronger structural clamping, smaller sampling steps, and longer training.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/proposal.json`
- Metrics: average_num_violations=1237.6666666666667, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=953.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_124104_6e7e95](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95)
- Next attempt: Run one exploit candidate: clamp all structural lambdas almost at 5.0 (node_count tightest), reduce sampling_step_size again, keep sparse masking essentially zero, and extend total_steps.

#### Files to inspect

- run: [artificial_graphs_large_20260628_124104_6e7e95](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_124104_6e7e95:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_124104_6e7e95:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_124104_6e7e95](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/proposal.json).
- Why: exploit around the previously improved high-constraint basin (tight lambdas near 5, reduced sampling step, longer training).
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/metrics/summary.csv): **average_num_violations=1237.6666666666667**, **median_num_violations=953.0**, **feasible_rate=0.0**; worse than prior rescue run.
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: keep strong structural bias but slightly relax lambda lower bounds, broaden sampling_step_size downward, and extend total_steps to 900–1250 for a more informative single-trial read.

#### Files to inspect

- run: [artificial_graphs_large_20260628_124104_6e7e95](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_124104_6e7e95/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_124104_6e7e95:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_134125_3e28dd:begin -->
### artificial_graphs - artificial_graphs_large_20260628_134125_3e28dd

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest single-trial exploit regressed from the previous best (violations rose from avg 604.33/median 557 to avg 1237.67/median 953; feasible_rate stayed 0.0). Next attempt should remain in the same high-structure regime but re-broaden key ranges to avoid over-tight local behavior and improve single-sample robustness.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/proposal.json`
- Metrics: average_num_violations=1403.2307692307693, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=1140.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_134125_3e28dd](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd)
- Next attempt: Run one re-broadened candidate: keep node_count lambda very high, slightly relax other lambda lower bounds, allow smaller sampling_step_size, and increase total_steps to 900-1250.

#### Files to inspect

- run: [artificial_graphs_large_20260628_134125_3e28dd](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_134125_3e28dd:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_134125_3e28dd:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_134125_3e28dd](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/proposal.json).
- Why: re-broaden around the prior high-structure setup after the previous regression.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/metrics/summary.csv): **average_num_violations=1403.2307692307693**, **median_num_violations=1140.0**, **feasible_rate=0.0** (worse than the prior run).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: keep `lambda_node_count_importance` very high, relax other lambda lower bounds modestly, widen `sampling_step_size` downward/upward, and increase `total_steps` to 1100–1500.

#### Files to inspect

- run: [artificial_graphs_large_20260628_134125_3e28dd](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_134125_3e28dd/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_134125_3e28dd:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_154148_fad88b:begin -->
### artificial_graphs - artificial_graphs_large_20260628_154148_fad88b

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest completed trial further regressed (avg violations 1403.23, median 1140.0, feasible_rate 0.0) versus recent attempts, suggesting the current ultra-tight near-5 lambda ranges are over-constrained and brittle with batch_size=1. Next step is a small, controlled re-broadening while preserving high node-count pressure and extending training budget.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/proposal.json`
- Metrics: average_num_violations=302.25, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=302.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_154148_fad88b](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b)
- Next attempt: Run one controlled re-broadened candidate: keep node_count lambda near 5.0, modestly lower other lambda floors, widen sampling_step_size, and train longer (1100-1500 steps).

#### Files to inspect

- run: [artificial_graphs_large_20260628_154148_fad88b](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_154148_fad88b:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_154148_fad88b:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_154148_fad88b](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/proposal.json).
- Why: recover from prior regression by modestly re-broadening lambda floors, keeping `lambda_node_count_importance` near 5.0, widening `sampling_step_size`, and extending training steps.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/metrics/summary.csv): **average_num_violations=302.25**, **median_num_violations=302.0**, **feasible_rate=0.0** (large improvement vs previous run, but still no feasible samples).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: locally refine around this successful region (especially `sampling_step_size` and lambda bands) and raise `total_steps` to 1300–1800 to test whether violations can be driven into feasible territory.

#### Files to inspect

- run: [artificial_graphs_large_20260628_154148_fad88b](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_154148_fad88b/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_154148_fad88b:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_174206_cdca59:begin -->
### artificial_graphs - artificial_graphs_large_20260628_174206_cdca59

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest completed trial improved sharply (avg violations 302.25, median 302.0) from the prior regression (1403.23, 1140.0), confirming the re-broadened near-5 lambda regime is much healthier. Since feasible_rate is still 0.0, the next attempt should exploit this basin with a tighter local search and slightly longer training.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/proposal.json`
- Metrics: average_num_violations=414.125, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=337.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_174206_cdca59](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59)
- Next attempt: Run one local-refinement candidate around the improved setting: keep node_count lambda near 5.0, center other lambdas near the successful draw, narrow sampling_step_size around ~3e-4 to 5.5e-4, and increase total_steps to 1300-1800.

#### Files to inspect

- run: [artificial_graphs_large_20260628_174206_cdca59](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_174206_cdca59:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_174206_cdca59:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_174206_cdca59](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/proposal.json).
- Why: refine the previously improved high-lambda basin with longer training and controlled sampling.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/metrics/summary.csv): **average_num_violations=414.125**, **median_num_violations=337.5**, **feasible_rate=0.0** (worse than prior best 302.25/302.0, but far better than earlier regression).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: keep all lambdas near the high regime (node_count closest to 5.0), tighten `sampling_step_size`, reduce sparse mask toward ~0, and increase `total_steps` to test whether extra optimization can recover/improve toward feasibility.

#### Files to inspect

- run: [artificial_graphs_large_20260628_174206_cdca59](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_174206_cdca59/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_174206_cdca59:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_214312_642c72:begin -->
### artificial_graphs - artificial_graphs_large_20260628_214312_642c72

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest mini-batch remained in the improved high-lambda region but regressed from the best observed violation level, suggesting variance within the basin rather than a directional failure. Next attempt should stay local, tighten sampling and regularization ranges, and raise total_steps so the same basin has more chance to convert lower violations into feasible samples.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/proposal.json`
- Metrics: average_num_violations=1225.0833333333333, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=930.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_214312_642c72](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72)
- Next attempt: Run one local-refinement candidate with tighter near-5 lambdas, narrower sampling_step_size (~2.8e-4 to 4.2e-4), near-zero sparse supervision mask, and longer training (1600-2200 total_steps).

#### Files to inspect

- run: [artificial_graphs_large_20260628_214312_642c72](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_214312_642c72:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_214312_642c72:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_214312_642c72](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/proposal.json).
- Why: local refinement around the previously improved high-lambda regime with tighter sampling and longer training.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/metrics/summary.csv): **average_num_violations=1225.0833**, **median_num_violations=930.5**, **feasible_rate=0.0** (sharp regression vs previous best basin performance).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: recovery-focused local search with lambdas shifted tighter toward 5.0, smaller `sampling_step_size`, near-zero sparse mask, slightly lower LR, and longer `total_steps` to test whether violations return to the prior lower basin.

#### Files to inspect

- run: [artificial_graphs_large_20260628_214312_642c72](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_214312_642c72/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_214312_642c72:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_234356_2433d8:begin -->
### artificial_graphs - artificial_graphs_large_20260628_234356_2433d8

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest run completed successfully but regressed heavily (avg violations 1225.08, median 930.5, feasible_rate 0.0) relative to the prior improved basin. The most informative next step is a conservative recovery refinement: keep architecture fixed, move all loss weights closer to 5.0, reduce sampling step size/noise exposure, and increase training steps.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/proposal.json`
- Metrics: average_num_violations=959.3333333333334, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=910.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260628_234356_2433d8](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8)
- Next attempt: Sample one recovery candidate in a tighter near-5 lambda regime with smaller sampling_step_size, near-zero sparse supervision mask, slightly lower learning rate, and longer training (2000-3000 steps).

#### Files to inspect

- run: [artificial_graphs_large_20260628_234356_2433d8](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260628_234356_2433d8:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260628_234356_2433d8:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260628_234356_2433d8](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/proposal.json).
- Why: recovery refinement after prior regression, keeping architecture fixed and searching near high-lambda settings.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/metrics/summary.csv): **average_num_violations=959.3333**, **median_num_violations=910.0**, **feasible_rate=0.0** (improved vs previous run, but still no feasible samples).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: another tight local recovery step with slightly longer training, slightly lower LR, smaller `sampling_step_size`, near-zero sparse mask, and all major lambdas concentrated even closer to 5.0.

#### Files to inspect

- run: [artificial_graphs_large_20260628_234356_2433d8](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260628_234356_2433d8/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260628_234356_2433d8:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_004410_cd34c4:begin -->
### artificial_graphs - artificial_graphs_large_20260629_004410_cd34c4

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest trial improved violations substantially (avg 959.33, median 910.0) compared with the previous regression, but feasible_rate is still 0.0. The next informative step is a tight local refinement in the same high-lambda basin with smaller sampling steps and longer training.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/proposal.json`
- Metrics: average_num_violations=542.5, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=377.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260629_004410_cd34c4](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4)
- Next attempt: Sample one candidate with longer total_steps (2600-4200), slightly lower learning rate, tighter near-5 lambda ranges (especially auxiliary/edge_count/degree-consistency/node-exist/node-label), and smaller sampling_step_size while keeping sparse supervision near zero.

#### Files to inspect

- run: [artificial_graphs_large_20260629_004410_cd34c4](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260629_004410_cd34c4:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_004410_cd34c4:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260629_004410_cd34c4](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/proposal.json).
- Why: continue local refinement in the high-lambda regime after prior recovery trend.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/metrics/summary.csv): **average_num_violations=542.5**, **median_num_violations=377.5**, **feasible_rate=0.0** (substantial violation reduction, still no feasible samples).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: keep architecture fixed; extend `total_steps`, bias `learning_rate` slightly lower, tighten weaker lambdas closer to 5.0, and reduce `sampling_step_size` slightly to test whether lower violations can cross into non-zero feasibility.

#### Files to inspect

- run: [artificial_graphs_large_20260629_004410_cd34c4](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_004410_cd34c4/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260629_004410_cd34c4:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_034501_005384:begin -->
### artificial_graphs - artificial_graphs_large_20260629_034501_005384

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest trial cut violations sharply (avg 542.5, median 377.5) but feasible_rate is still 0.0; continue a tight local search around this improved high-lambda region with longer training and slightly smaller sampling steps to target first feasible samples.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/proposal.json`
- Metrics: average_num_violations=1274.375, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=1010.5, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260629_034501_005384](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384)
- Next attempt: Sample one candidate with total_steps shifted upward (3200-5600), a slightly lower learning-rate band, weaker lambdas tightened to >=4.99, and slightly smaller sampling_step_size while keeping sparse supervision near zero.

#### Files to inspect

- run: [artificial_graphs_large_20260629_034501_005384](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260629_034501_005384:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_034501_005384:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260629_034501_005384](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/proposal.json).
- Why: continue local refinement after the earlier strong violation drop.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/metrics/summary.csv): **average_num_violations=1274.375**, **median_num_violations=1010.5**, **feasible_rate=0.0** (clear regression vs prior best; still no feasible samples).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: run a recovery-biased sample with tighter near-5 lambdas, slightly longer `total_steps`, and smaller `sampling_step_size` while keeping architecture fixed.

#### Files to inspect

- run: [artificial_graphs_large_20260629_034501_005384](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_034501_005384/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260629_034501_005384:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_054528_5eda26:begin -->
### artificial_graphs - artificial_graphs_large_20260629_054528_5eda26

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26 finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest mini-batch regressed (avg/median violations increased to 1274.375/1010.5) and feasible_rate remains 0.0, so we should recover by tightening into a stricter high-lambda basin, extending training steps, and slightly reducing sampling step size.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/proposal.json`
- Metrics: average_num_violations=620.875, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=578.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260629_054528_5eda26](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26)
- Next attempt: Sample one candidate from a recovery-focused narrow range: keep architecture fixed, push total_steps higher, tighten all lambdas closer to 5.0, and reduce sampling_step_size to recover lower violations and target first feasible samples.

#### Files to inspect

- run: [artificial_graphs_large_20260629_054528_5eda26](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260629_054528_5eda26:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_054528_5eda26:agent:begin -->
- Tried 1 `range_search` candidate in run [artificial_graphs_large_20260629_054528_5eda26](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26) from [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/proposal.json).
- Why: recovery-focused local search after prior regression, with high lambdas and reduced sampling step size.
- Result from [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/metrics/summary.csv): **average_num_violations=620.875**, **median_num_violations=578.0**, **feasible_rate=0.0** (clear improvement in violations, still no feasible samples).
- Artifacts: [trial config](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/config.yaml), [trial log](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/logs/trial_001.log), [trial results](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/trial_results.csv), [loss PDF](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/trials/trial_001/metrics/loss_curves.pdf).
- Next proposed attempt: keep architecture fixed, tighten around near-5 lambdas, reduce `sampling_step_size` slightly further, and raise `total_steps` to test conversion of lower violations into first feasible samples.

#### Files to inspect

- run: [artificial_graphs_large_20260629_054528_5eda26](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_054528_5eda26/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260629_054528_5eda26:agent:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_074601_8a977b:begin -->
### artificial_graphs - artificial_graphs_large_20260629_074601_8a977b

- Tried: 1 candidate(s) from `range_search`.
- Agent reasoning: Latest run /run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b finished with no metrics. Proposing the next mini-batch from the configured mutable ranges and exact patches. Latest mini-batch improved substantially (average/median violations 620.875/578.0 from 1274.375/1010.5) but feasible_rate remains 0.0; we should exploit this better region with longer training and slightly smaller sampling steps while keeping high structural penalties.
- Mutable groups: architecture, training, loss_weights, sampling
- Ranges/exact configs: `/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/proposal.json`
- Metrics: average_num_violations=769.8666666666667, campaign_trial_id=1, feasible_rate=0.0, loss_pdf_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics/loss_curves.pdf, median_num_violations=517.0, results_csv_path=/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/NodeField/artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics/trial_results.csv, trial_id=1
- Artifacts: [artificial_graphs_large_20260629_074601_8a977b](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b)
- Next attempt: Sample one candidate from a narrow exploitative range around the improved run: keep architecture fixed, increase total_steps, keep lambdas very close to 5.0, and lower sampling_step_size to target first feasible outputs.

#### Files to inspect

- run: [artificial_graphs_large_20260629_074601_8a977b](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/campaign_result.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/logs/mini_batch.log)
- trial_001 log: [trial_001.log](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/logs/trial_001.log)
- trial_001 loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics/loss_curves.pdf)
- trial_001 config: [config.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/config.yaml)
- trial_001 metrics: [metrics.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics.json)
- trial_001 results CSV: [trial_results.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics/trial_results.csv)
- loss PDF: [loss_curves.pdf](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/trials/trial_001/metrics/loss_curves.pdf)
<!-- nodefield-campaign:artificial_graphs_large_20260629_074601_8a977b:end -->

<!-- nodefield-campaign:artificial_graphs_large_20260629_074601_8a977b:agent:begin -->
The run finished successfully at the trial level and produced metrics, but it did not generate any feasible samples yet. Compared with the previous strong run, the median violations improved, while the average violations worsened, which suggests some progress on typical cases but persistent high-violation tails.

| trial | average_num_violations | median_num_violations | feasible_rate | note |
|---|---:|---:|---:|---|
| trial_001 | 769.8667 | 517.0 | 0.0 | Completed; no feasible samples |

Next, we should keep the same architecture and near-max structural penalties, but tighten the search locally with a lower sampling step size and more total training steps to improve consistency and try to convert lower median violations into first feasible outputs.

#### Files to inspect

- run: [artificial_graphs_large_20260629_074601_8a977b](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b)
- state: [state.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/state.json)
- proposal: [proposal.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/proposal.json)
- campaign result: [campaign_result.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/campaign_result.json)
- agent decision: [agent_decision.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/agent_decision.json)
- metrics csv: [summary.csv](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/metrics/summary.csv)
- metrics json: [summary.json](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/metrics/summary.json)
- campaign config: [campaign.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/configs/campaign.yaml)
- base workflow: [base_workflow.yaml](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/configs/base_workflow.yaml)
- mini-batch log: [mini_batch.log](artifact/artificial_graphs/artificial_graphs_large_20260629_074601_8a977b/logs/mini_batch.log)
<!-- nodefield-campaign:artificial_graphs_large_20260629_074601_8a977b:agent:end -->
