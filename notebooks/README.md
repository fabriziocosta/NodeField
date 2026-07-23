# NodeField notebooks

Each notebook below is linked to its `.ipynb` file so it can be opened directly
from VS Code. Start with the notebook that matches the task rather than treating
these as one linear tutorial.

## Core workflows

- [Molecular generation and guidance](./molecular_generation_and_guidance.ipynb) — maintained chemistry workflow for molecular training, decoding, feasibility, guidance, and interpolation.
- [ZINC molecular-generation development](./zinc_molecular_generation_development.ipynb) — larger cached-ZINC development and evaluation workspace.
- [Artificial non-streaming training and sampling](./artificial_non_streaming_training_and_sampling.ipynb) — train on generated cycle/path/star graphs and inspect samples.

## Artificial graph experiments

- [Artificial partial-graph token reconstruction](./artificial_partial_graph_token_reconstruction.ipynb) — reconstruct full artificial graphs from partial graph token conditioning.
- [Artificial structural conditioning analysis](./artificial_structural_conditioning_analysis.ipynb) — test whether path, cycle, and ray conditioning steers generated graph structure.
- [Sample latest artificial generator](./sample_latest_artificial_generator.ipynb) — load the newest saved artificial generator and inspect generated samples and feasibility.

## Molecular training and sampling

- [ZINC non-streaming training and sampling](./zinc_non_streaming_training_and_sampling.ipynb) — materialize a ZINC subset, train in memory, and compare filtered/unfiltered samples.
- [ZINC streaming training and sampling](./zinc_streaming_training_and_sampling.ipynb) — train directly from the ZINC CSV through the streaming path.
- [Campaign best-trial sampling review](./campaign_best_trial_sampling_review.ipynb) — reload the best completed campaign trial and review its outputs.

## Optimization and feasibility studies

- [Similarity-pruned target optimization](./similarity_pruned_target_optimization.ipynb) — train and evaluate a generator optimized toward a hidden similarity target.
- [ZINC molecule hyperparameter optimization](./zinc_molecule_hyperparameter_optimization.ipynb) — run the YAML-driven ZINC molecule hyperparameter search.
- [ZINC feasibility oracle analysis](./zinc_feasibility_oracle_analysis.ipynb) — compare oracle-off and oracle-on decoding, traces, and interpolation.
- [ZINC guidance bootstrap and cycle analysis](./zinc_guidance_bootstrap_and_cycle_analysis.ipynb) — bootstrap guidance and compare guided versus unguided sampling cycles.

## Data preparation and validation

- [Filter ZINC molecules by node count](./filter_zinc_molecules_by_node_count.ipynb) — create ZINC CSV subsets by molecular graph size.
- [Initialize notebook environment](./initialize_notebook_environment.ipynb) — prepare the kernel, repository paths, and optional NSPPK dependency.
- [Validate node-order equivariance](./validate_node_order_equivariance.ipynb) — verify permutation consistency through the NodeField data and encoder paths.

Generated artificial dataset configs are kept in
[configs/artificial_datasets](./configs/artificial_datasets), rather than in
this folder’s common space. For the longer documentation catalog, see
[../docs/NOTEBOOKS.md](../docs/NOTEBOOKS.md).
