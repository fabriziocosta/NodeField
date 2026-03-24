import json
from pathlib import Path

from conditional_node_field_graph_generator.notebooks import configure_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
MAINTAINED_NOTEBOOKS = [
    NOTEBOOK_ROOT / "demo.ipynb",
    NOTEBOOK_ROOT / "demo_chem.ipynb",
    NOTEBOOK_ROOT / "demo_optimization.ipynb",
    NOTEBOOK_ROOT / "demo_zinc.ipynb",
    NOTEBOOK_ROOT / "demo_zinc_guidance_bootstrap.ipynb",
    NOTEBOOK_ROOT / "demo_zinc_hyperparameter_search.ipynb",
    NOTEBOOK_ROOT / "demo_zinc_oracle_study.ipynb",
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
        assert "from _notebook_bootstrap import configure_notebook" not in source
        assert "from conditional_node_field_graph_generator.notebooks import configure_notebook" in source


def test_compatibility_notebook_bootstrap_reexports_package_helpers():
    from _notebook_bootstrap import configure_notebook as shim_configure_notebook

    assert shim_configure_notebook is configure_notebook
