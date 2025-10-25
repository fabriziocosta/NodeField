# GraphGen

GraphGen is an experimental playground for conditional graph generation. It blends diffusion-style denoising with transformer cross-attention to produce node-level graph representations that respect structural constraints such as node existence, degree, and (optionally) edge labels. The project is organised as a small library for training/generating models plus an exploratory notebook that ties everything together.

## Highlights
- Cross-attention transformer backbone that conditions node embeddings on graph-level tokens and diffusion time embeddings.
- Diffusion objective with auxiliary heads for node existence and discrete degree prediction, keeping generated graphs structurally plausible.
- Optional edge supervision via a lightweight MLP head that learns pairwise connectivity when labelled edges are provided.
- Classifier-guided sampling and metric logging utilities to steer the model toward desired graph attributes.
- Low-rank MLP components for classical tasks (classification/regression) that can be reused inside broader pipelines.

## Repository Layout
- `node_diffusion/conditional_denoising_node_generator.py` – core LightningModule, training loop, sampler, dataset helpers, and the scikit-learn style `ConditionalNodeGenerator` facade.
- `node_diffusion/low_rank_mlp.py` – reusable low-rank MLP blocks, Lightning wrappers, and callbacks.
- `node_diffusion/decompositional_encoder_decoder.py` – graph encoder/decoder utilities and optimisation helpers (requires the separate `coco_grape` package).
- `notebooks/Denoising Conditional Node Graph Generator.ipynb` – end-to-end workflow that prepares data, trains the diffusion model, and visualises generations.

## Installation
1. Create and activate a Python environment (Python ≥ 3.9 recommended).
2. Install the core dependencies:
   ```bash
   pip install "numpy<2" torch pytorch-lightning scipy pandas scikit-learn networkx matplotlib pulp dill
   ```
3. Install project-specific extras:
   - `coco-grape` (available from the companion repository) for data processing helpers referenced in `decompositional_encoder_decoder.py`.
   - Optional: `jupyterlab` or `notebook` if you want to run the provided notebook.

GPU support is recommended for larger experiments but the code paths also run on CPU.

## Usage

### 1. Prepare your data
- `node_encodings_list`: list of `(num_nodes, num_features)` arrays. Column `0` is treated as a binary existence flag; column `1` is a (possibly scaled) node degree. Remaining columns can hold arbitrary continuous features.
- `conditional_graph_encodings`: array of `(num_graphs, condition_dim)` graph-level descriptors the diffusion model will condition on.
- Optional `edge_pairs`/`edge_targets`: tuples `(graph_index, i, j)` plus edge labels if you want the edge supervision head to learn connectivity.
- Optional `node_mask`: boolean mask marking valid (unpadded) nodes per graph.

Ensure the order of graphs matches across these collections.

### 2. Initialise, set up, and train
```python
from node_diffusion.conditional_denoising_node_generator import ConditionalNodeGenerator

generator = ConditionalNodeGenerator(
    maximum_epochs=100,
    batch_size=32,
    total_steps=500,
    latent_embedding_dimension=128,
    verbose=True,
    use_guidance=True,  # enable optional classifier guidance
)

generator.setup(
    node_encodings_list,
    conditional_graph_encodings,
    edge_pairs=edge_pairs,
    edge_targets=edge_targets,
    node_mask=node_mask,
)

generator.fit(
    node_encodings_list,
    conditional_graph_encodings,
    edge_pairs=edge_pairs,
    edge_targets=edge_targets,
    node_mask=node_mask,
)
```
`setup` performs scaling, padding, and model initialisation; `fit` launches the PyTorch Lightning trainer (90/10 train/validation split by default).

### 3. Sample new graphs
```python
samples = generator.predict(conditional_graph_encodings[:4])
```
`predict` returns denoised node feature tensors in the original scale. When edge supervision is enabled the diffusion sampler also projects node existence and degree channels using the auxiliary heads.

> Notebook hint: the repo modules sit outside `notebooks/`, so either launch Jupyter from the project root or prepend the repository directory to `sys.path` (the notebook includes a helper cell that does this automatically).

### 4. Optional classifier guidance
After training you can steer sampling toward specific graph classes:
```python
generator.set_guidance_classifier(num_classes=target_labels.max() + 1)
generator.train_guidance_classifier(node_feats, cond_vecs, target_labels, epochs=20, lr=1e-3)
guided = generator.predict(cond_vecs, desired_class=2)
```
`node_feats` and `cond_vecs` should use the scaled representations stored during training (see the notebook for an end-to-end example).

### 5. Inspect training metrics
```python
generator.plot_metrics(window=10, alpha=0.3)
```
This overlays raw and smoothed losses (total, reconstruction, existence, degree, and edge metrics when enabled).

## Notebook Workflow
The notebook in `notebooks/` demonstrates how the diffusion generator hooks into a broader pipeline: loading graphs, vectorising them with the `DecompositionalNodeEncoderDecoder`, training the diffusion model, performing graph interpolation, and augmenting labelled datasets. Launch it with `jupyter lab notebooks/Denoising Conditional Node Graph Generator.ipynb` after installing the dependencies.

## Status
This codebase is research-oriented and still evolving. Expect rapid changes, limited error handling, and a dependence on the surrounding `coco-grape` ecosystem. Contributions, issues, and experiment notes are welcome.

## Troubleshooting
- **NumPy / PyTorch mismatch** – Some prebuilt PyTorch wheels are compiled against NumPy 1.x. If you hit `_ARRAY_API not found` or similar import errors, ensure you install `numpy<2` (as shown above) or upgrade to a PyTorch build that officially supports NumPy ≥2.
- **Degree collapse during sampling** – The sampler snaps the degree channel to the classifier head’s argmax. If you need the raw diffusion output instead, disable `use_heads_projection` when calling `ConditionalNodeGenerator.predict` or improve/temper the auxiliary classifier.
