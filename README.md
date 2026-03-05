# GraphGen

GraphGen is a Python toolkit for conditional graph generation with decompositional encoding/decoding and equilibrium-style node updates.

The repository includes:
- A trainable conditional node generator.
- A graph-level generator that handles encoding, supervision construction, and decoding.
- Notebook workflows for experiments and analysis.
- Unit tests for core utility and generation behavior.

## Project Layout

- `eqm_decompositional_graph_generator/`
  Core package:
  - `eqm_conditional_node_generator.py`: model, training loop integration, sampling utilities.
  - `decompositional_encoder_decoder.py`: graph generator and decoder orchestration.
  - `graph_generator.py`: public graph generator entrypoints.
  - `generator_shared.py`, `lightning_utils.py`, `low_rank_mlp.py`: shared modules.
- `notebooks/`
  Experiment and demo notebooks, plus notebook-specific helpers in `notebook_utils.py`.
- `tests/`
  Pytest suite for generator behavior and helper modules.
- `.artifacts/`
  Local artifacts (checkpoints/models); ignored by git.

## Installation

1. Create a Python environment (Python 3.10+ recommended).
2. Install core dependencies:

```bash
pip install "numpy<2" torch pytorch-lightning scipy pandas scikit-learn networkx matplotlib pulp dill
```

3. Install optional extras as needed:
- `jupyterlab` or `notebook` to run notebooks.
- `coco-grape`, `NSPPK`, `AbstractGraph` if you run notebook flows that import them.

## Quick Start

```python
from eqm_decompositional_graph_generator import (
    EqMDecompositionalNodeGenerator,
    EqMDecompositionalGraphDecoder,
    EqMDecompositionalGraphGenerator,
)
```

Typical high-level workflow:
1. Prepare graphs (`networkx.Graph`) with node/edge labels as needed.
2. Build vectorizers for graph-level and node-level embeddings.
3. Instantiate `EqMDecompositionalNodeGenerator`.
4. Wrap it in `EqMDecompositionalGraphGenerator` (optionally with a decoder).
5. Train with `.fit(...)`.
6. Generate with `.sample(...)` or `.sample_conditioned_on_random(...)`.

Notebook examples:
- `notebooks/demo.ipynb`
- `notebooks/demo2.ipynb`
- `notebooks/demo_chem.ipynb`
- `notebooks/demo_chem2.ipynb`
- `notebooks/demo_optimization.ipynb`

## Running Tests

```bash
pytest -q
```

Targeted run example:

```bash
pytest tests/test_graph_generator.py -q
```

## Data and Artifacts

Large datasets and training artifacts are intentionally excluded from version control.

Ignored locations include:
- `.artifacts/`
- `PUBCHEM/`
- `notebooks/PUBCHEM/`

Keep experimental outputs in ignored paths to avoid inflating repository history.

## Notes for Notebook Development

Notebook execution flow is kept lean by design:
- Prefer assigning variables and calling functions from `.py` modules.
- Place reusable notebook logic in helper modules (for example `notebooks/notebook_utils.py`).
- Clear notebook outputs before committing.
