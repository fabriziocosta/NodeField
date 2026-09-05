# NodeField notebooks

Start with [setup.ipynb](setup.ipynb) to install dependencies. Restart the kernel afterward,
then select the same Python environment in the notebook you want to run.

Choose a task below. Each notebook explains its prerequisites, settings, next steps,
and outputs. Review data sizes and training budgets before running training cells.

## Synthetic graphs

| Task | Before you start | Results |
| --- | --- | --- |
| [Train a synthetic graph model](synthetic/train.ipynb) | NSPPK; generated data | Saved model, dataset configuration, and sample plots |
| [Sample a synthetic graph model](synthetic/sample.ipynb) | Saved synthetic model and dataset configuration | Sample plots and feasibility comparisons |
| [Reconstruct partial graphs](synthetic/reconstruct.ipynb) | NSPPK; generated data | Reconstruction comparisons |
| [Evaluate structural conditioning](synthetic/evaluate_conditioning.ipynb) | Saved synthetic model and dataset configuration | Structure and label comparisons |

## Molecules

| Task | Before you start | Results |
| --- | --- | --- |
| [Generate and compare molecules](molecular/generate.ipynb) | NSPPK, RDKit, and access to PubChem data | Trained model, molecule samples, and interpolation plots |
| [Train on ZINC in memory](molecular/train_zinc.ipynb) | NSPPK, RDKit, and a ZINC CSV; sufficient RAM for the selected data | Model and training progress artifacts |
| [Train on a ZINC stream](molecular/train_zinc_streaming.ipynb) | NSPPK, RDKit, and the selected ZINC CSV | Model, streaming statistics, and sample plots |
| [Prepare ZINC data](molecular/prepare_zinc.ipynb) | RDKit; network access if the source data is missing | Filtered ZINC CSV files |
| [Evaluate molecular feasibility](molecular/evaluate_feasibility.ipynb) | Saved molecular model and matching ZINC data | Decode traces, score comparisons, and interpolation plots |

## Campaigns

| Task | Before you start | Results |
| --- | --- | --- |
| [Tune a ZINC model](campaigns/tune_zinc.ipynb) | ZINC data, molecular dependencies, and a search configuration | Trial results, best model, and sample comparisons |
| [Monitor a campaign](campaigns/monitor.ipynb) | Existing campaign state; Graphviz support for relationship diagrams | Interactive campaign dashboard |
| [Review the best campaign trial](campaigns/review_best_trial.ipynb) | Completed campaign trials and saved model artifacts | Trial ranking and generated molecule review |

## Experiments

| Task | Before you start | Results |
| --- | --- | --- |
| [Experiment with guidance cycles](experiments/guidance_cycles.ipynb) | NSPPK, RDKit, ZINC data, and a saved model or training budget | Guidance cycle summaries and sample comparisons |
| [Experiment with ZINC generation](experiments/zinc_generation.ipynb) | NSPPK, RDKit, and ZINC data | Development model, sampling and interpolation comparisons |
| [Optimize graph similarity](experiments/target_similarity.ipynb) | NSPPK, RDKit, and access to the selected dataset | Saved model and similarity-guidance comparisons |

## Validation

| Task | Before you start | Results |
| --- | --- | --- |
| [Validate node-order equivariance](validation/node_order_equivariance.ipynb) | NSPPK; generated data | Permutation checks and diagnostic output |

## Files and conventions

- `configs/` contains shared workflow settings; generated synthetic configurations go in `configs/artificial_datasets/`.
- `datasets/` contains input data. Moving a notebook does not move its data.
- `.artifacts/` at the project root contains models and training outputs; campaigns use `artifact/`.
- Run notebooks in order within each file. There is no required order across workflows except their stated prerequisites.
- Keep reusable computation and plotting helpers in the package. Keep notebook settings and explanations near the steps they control.
- Clear execution outputs before committing. Save generated results in the artifact folders.

The in-memory and streaming ZINC training notebooks remain separate because their
model settings, preprocessing, and training interfaces differ. Sampling a saved
synthetic model has its own notebook so it does not require training again.
