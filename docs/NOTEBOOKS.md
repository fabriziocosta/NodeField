# Notebook guide

Notebook filenames describe the workflow they implement. The repository does
not use a generic `demo_` prefix because the useful distinction is whether a
notebook trains a model, samples from one, analyzes an experiment, prepares
data, or validates an invariant.

## Start here

- [Molecular generation and guidance](../notebooks/molecular_generation_and_guidance.ipynb)
  is the maintained chemistry-focused workflow.
- [ZINC molecular-generation development](../notebooks/zinc_molecular_generation_development.ipynb)
  is the maintained larger ZINC development workflow.
- [Artificial non-streaming training and sampling](../notebooks/artificial_non_streaming_training_and_sampling.ipynb)
  is the smallest artificial training path.

## Artificial graph experiments

| Notebook | Purpose |
| --- | --- |
| [Artificial non-streaming training and sampling](../notebooks/artificial_non_streaming_training_and_sampling.ipynb) | Train on generated cycle/path/star graphs and inspect unfiltered and feasibility-filtered samples. |
| [Artificial partial-graph token reconstruction](../notebooks/artificial_partial_graph_token_reconstruction.ipynb) | Remove nodes from artificial graphs, use the remaining graph as token conditioning, and reconstruct the full graph. |
| [Artificial structural conditioning analysis](../notebooks/artificial_structural_conditioning_analysis.ipynb) | Test whether structural conditioning steers generated artificial graphs toward path, cycle, or ray statistics. |
| [Sample latest artificial generator](../notebooks/sample_latest_artificial_generator.ipynb) | Load the newest saved artificial generator and inspect generated samples and feasibility estimates. |

## Molecular generation and analysis

| Notebook | Purpose |
| --- | --- |
| [Molecular generation and guidance](../notebooks/molecular_generation_and_guidance.ipynb) | Train and inspect a chemistry-oriented generator, including masks, bond labels, feasibility filtering, guidance, and interpolation. |
| [ZINC molecular-generation development](../notebooks/zinc_molecular_generation_development.ipynb) | Develop and evaluate the cached ZINC workflow with conditional sampling, interpolation, and guidance. |
| [ZINC non-streaming training and sampling](../notebooks/zinc_non_streaming_training_and_sampling.ipynb) | Materialize a ZINC subset, train through the regular in-memory path, and sample molecules. |
| [ZINC streaming training and sampling](../notebooks/zinc_streaming_training_and_sampling.ipynb) | Train from the ZINC CSV through the streaming path and sample molecules. |
| [Campaign best-trial sampling review](../notebooks/campaign_best_trial_sampling_review.ipynb) | Select the best completed campaign trial, reload its checkpoint, and review training and generated samples. |

## ZINC optimization and feasibility studies

| Notebook | Purpose |
| --- | --- |
| [ZINC molecule hyperparameter optimization](../notebooks/zinc_molecule_hyperparameter_optimization.ipynb) | Run the YAML-driven molecule hyperparameter search and review its best trial. |
| [ZINC feasibility oracle analysis](../notebooks/zinc_feasibility_oracle_analysis.ipynb) | Compare oracle-off and oracle-on decoding, traces, feasibility scores, and interpolation. |
| [ZINC guidance bootstrap and cycle analysis](../notebooks/zinc_guidance_bootstrap_and_cycle_analysis.ipynb) | Bootstrap regression guidance from generated molecules and compare guided versus unguided cycles. |
| [Similarity-pruned target optimization](../notebooks/similarity_pruned_target_optimization.ipynb) | Prune a dataset by graph similarity to a hidden target, train an optimization model, and test classifier guidance. |

## Data preparation and validation

| Notebook | Purpose |
| --- | --- |
| [Filter ZINC molecules by node count](../notebooks/filter_zinc_molecules_by_node_count.ipynb) | Download ZINC and write CSV subsets by molecular graph node count. |
| [Initialize notebook environment](../notebooks/initialize_notebook_environment.ipynb) | Prepare the local notebook/kernel environment and optional NSPPK checkout. |
| [Validate node-order equivariance](../notebooks/validate_node_order_equivariance.ipynb) | Check that graph rows, targets, adjacency, conditioning, and the transformer path respect node permutations. |

Notebook configuration files remain under [`notebooks/configs`](../notebooks/configs).
Generated artificial dataset configs specifically belong in
[`notebooks/configs/artificial_datasets`](../notebooks/configs/artificial_datasets),
not directly in the notebook root. The directory is ignored so generated YAML
does not pollute the repository or the common notebook space.
Large datasets, checkpoints, and generated outputs are kept under ignored
artifact/data paths rather than committed with the notebooks.
