import ast
import json

import pytest
from pathlib import Path

from conditional_node_field_graph_generator.notebooks import configure_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
MAINTAINED_NOTEBOOKS = [
    NOTEBOOK_ROOT / "campaigns/monitor.ipynb",
    NOTEBOOK_ROOT / "synthetic/train.ipynb",
    NOTEBOOK_ROOT / "synthetic/reconstruct.ipynb",
    NOTEBOOK_ROOT / "synthetic/evaluate_conditioning.ipynb",
    NOTEBOOK_ROOT / "molecular/generate.ipynb",
    NOTEBOOK_ROOT / "experiments/target_similarity.ipynb",
    NOTEBOOK_ROOT / "molecular/train_zinc.ipynb",
    NOTEBOOK_ROOT / "molecular/train_zinc_streaming.ipynb",
    NOTEBOOK_ROOT / "experiments/guidance_cycles.ipynb",
    NOTEBOOK_ROOT / "molecular/evaluate_feasibility.ipynb",
    NOTEBOOK_ROOT / "campaigns/tune_zinc.ipynb",
    NOTEBOOK_ROOT / "experiments/zinc_generation.ipynb",
    NOTEBOOK_ROOT / "campaigns/review_best_trial.ipynb",
    NOTEBOOK_ROOT / "synthetic/sample.ipynb",
    NOTEBOOK_ROOT / "validation/node_order_equivariance.ipynb",
    NOTEBOOK_ROOT / "molecular/prepare_zinc.ipynb",
]


@pytest.mark.parametrize("folder", ["", "synthetic", "molecular", "campaigns", "experiments", "validation"])
def test_configure_notebook_resolves_paths_from_notebooks_cwd(monkeypatch, folder):
    monkeypatch.chdir(NOTEBOOK_ROOT / folder)

    context = configure_notebook(require_nsppk=False, print_torch=False)

    assert context["REPO_ROOT"] == PROJECT_ROOT
    assert context["ARTIFACT_ROOT"] == PROJECT_ROOT / ".artifacts"
    assert context["NOTEBOOK_DATA_ROOT"] == NOTEBOOK_ROOT / "datasets"
    assert context["CHECKPOINT_ROOT"] == PROJECT_ROOT / ".artifacts" / "checkpoints" / "node_field"
    assert context["SAVED_GENERATOR_ROOT"] == PROJECT_ROOT / ".artifacts" / "saved_generators"


def test_maintained_notebooks_use_standardized_bootstrap_and_no_inline_path_probes():
    for notebook_path in MAINTAINED_NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "for _root in [Path.cwd(), *Path.cwd().parents]:" not in source
        assert "next(\n    candidate.resolve()" not in source
        assert "_repo_root = Path.cwd().resolve()" not in source
        assert "sys.path.insert(0" not in source
        assert "_notebook_bootstrap" not in source
        assert "from conditional_node_field_graph_generator.notebooks import configure_notebook" in source


def test_molecule_script_uses_same_config_and_helper_as_notebook():
    notebook = json.loads((NOTEBOOK_ROOT / "campaigns/tune_zinc.ipynb").read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    script_source = (SCRIPT_ROOT / "run_molecule_hyperparameter_optimization.py").read_text()

    assert "zinc_molecule_hyperparameter_optimization.yaml" in notebook_source
    assert "zinc_molecule_hyperparameter_optimization.yaml" in script_source
    assert "run_zinc_hyperparameter_optimization" in notebook_source
    assert "run_zinc_hyperparameter_optimization" in script_source


def test_maintained_notebooks_are_thin_and_use_named_feasibility_policies():
    legacy_terms = (
        "apply_feasibility_filtering",
        "use_feasibility_oracle",
        "feasibility_oracle_candidates_per_attempt",
    )
    for notebook_path in MAINTAINED_NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text())
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        assert all(term not in source for term in legacy_terms), notebook_path
        assert all(not cell.get("outputs") for cell in code_cells), notebook_path
        assert "from conditional_node_field_graph_generator.notebooks import configure_notebook" in source


def test_notebook_cells_have_valid_python_and_no_saved_outputs():
    for path in [NOTEBOOK_ROOT / "setup.ipynb", *MAINTAINED_NOTEBOOKS]:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Validate the timed cell body too, rather than only IPython's wrapper string.
            python_source = "\n".join(
                line for line in source.splitlines() if not line.startswith(("%", "!"))
            )
            ast.parse(python_source, filename=f"{path}:cell-{index}")
            assert cell["execution_count"] is None
            assert not cell["outputs"]


def test_shared_molecule_summary_handles_empty_and_labeled_graphs(capsys):
    import networkx as nx
    from conditional_node_field_graph_generator.extensions.demo.molecular_inspection import (
        label_counter,
        summarize_graphs,
    )

    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="O")
    graph.add_edge(0, 1, label="single")
    assert label_counter([graph]) == {"C": 1, "O": 1}
    assert label_counter([graph], "edge") == {"single": 1}
    summarize_graphs([graph], targets=[1], prefix="example")
    assert "2/2/2" in capsys.readouterr().out
    summarize_graphs([], prefix="empty")
    assert "empty: 0 graphs" in capsys.readouterr().out
