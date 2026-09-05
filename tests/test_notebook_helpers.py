import json
from pathlib import Path

from conditional_node_field_graph_generator.notebooks import configure_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
MAINTAINED_NOTEBOOKS = [
    NOTEBOOK_ROOT / "automatic_campaign_state_dashboard.ipynb",
    NOTEBOOK_ROOT / "artificial_non_streaming_training_and_sampling.ipynb",
    NOTEBOOK_ROOT / "artificial_partial_graph_token_reconstruction.ipynb",
    NOTEBOOK_ROOT / "artificial_structural_conditioning_analysis.ipynb",
    NOTEBOOK_ROOT / "molecular_generation_and_guidance.ipynb",
    NOTEBOOK_ROOT / "similarity_pruned_target_optimization.ipynb",
    NOTEBOOK_ROOT / "zinc_non_streaming_training_and_sampling.ipynb",
    NOTEBOOK_ROOT / "zinc_streaming_training_and_sampling.ipynb",
    NOTEBOOK_ROOT / "zinc_guidance_bootstrap_and_cycle_analysis.ipynb",
    NOTEBOOK_ROOT / "zinc_feasibility_oracle_analysis.ipynb",
    NOTEBOOK_ROOT / "zinc_molecule_hyperparameter_optimization.ipynb",
    NOTEBOOK_ROOT / "zinc_molecular_generation_development.ipynb",
    NOTEBOOK_ROOT / "campaign_best_trial_sampling_review.ipynb",
    NOTEBOOK_ROOT / "sample_latest_artificial_generator.ipynb",
    NOTEBOOK_ROOT / "validate_node_order_equivariance.ipynb",
    NOTEBOOK_ROOT / "filter_zinc_molecules_by_node_count.ipynb",
]


def test_configure_notebook_resolves_paths_from_notebooks_cwd(monkeypatch):
    monkeypatch.chdir(NOTEBOOK_ROOT)

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
    notebook = json.loads((NOTEBOOK_ROOT / "zinc_molecule_hyperparameter_optimization.ipynb").read_text())
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
