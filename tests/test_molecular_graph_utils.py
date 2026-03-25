from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from abstractgraph_graphicalizer.chem import (
    PubChemAssayLoader,
    SupervisedDataSetLoader,
    build_zinc_graph_corpus,
    extract_zinc_targets,
    graph_to_rdmol,
    load_zinc_graph_dataset,
    smiles_to_graph,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBCHEM_DATA_ROOT = PROJECT_ROOT / "notebooks" / "datasets" / "PUBCHEM"


def test_smiles_round_trip_to_rdkit():
    graph = smiles_to_graph("CCO")
    graph.graph["zinc_id"] = "z1"
    graph.graph["qed"] = 0.5

    assert graph is not None
    assert graph.graph["zinc_id"] == "z1"
    assert graph.graph["qed"] == 0.5
    assert graph.number_of_nodes() == 3

    mol = graph_to_rdmol(graph)
    assert mol.GetNumAtoms() == 3


def test_local_pubchem_loader_reads_bundled_sdf_files():
    loader = PubChemAssayLoader(PUBCHEM_DATA_ROOT)

    graphs, targets = loader.load("651610")

    assert len(graphs) == len(targets)
    assert len(graphs) > 100
    assert set(np.unique(np.asarray(targets))).issubset({0, 1})


def test_supervised_dataset_loader_equalizes_and_resizes():
    loader = PubChemAssayLoader(PUBCHEM_DATA_ROOT)

    def load_func():
        return loader.load("651610")

    graphs, targets = SupervisedDataSetLoader(
        load_func=load_func,
        size=40,
        use_equalized=True,
    ).load()

    assert len(graphs) == 40
    unique_targets, counts = np.unique(np.asarray(targets), return_counts=True)
    assert set(unique_targets.tolist()) == {0, 1}
    assert counts.tolist() == [20, 20]


def test_build_zinc_graph_corpus_from_tiny_csv(tmp_path):
    frame = pd.DataFrame(
        [
            {"zinc_id": "z1", "smiles": "CCO", "logP": 1.0, "qed": 0.5, "SAS": 2.0},
            {"zinc_id": "z2", "smiles": "c1ccccc1", "logP": 2.0, "qed": 0.7, "SAS": 1.5},
        ]
    )
    csv_path = tmp_path / "zinc_small.csv"
    frame.to_csv(csv_path, index=False)

    manifest = build_zinc_graph_corpus(tmp_path, csv_path)

    assert manifest["total_graphs"] == 2
    assert manifest["invalid_smiles_count"] == 0

    metadata = pd.DataFrame(frame)
    targets = extract_zinc_targets(metadata)
    assert targets.columns.tolist() == ["logP", "qed", "SAS"]


def test_load_zinc_graph_dataset_repairs_legacy_manifest(tmp_path):
    frame = pd.DataFrame(
        [
            {"zinc_id": "z1", "smiles": "CCO", "logP": 1.0, "qed": 0.5, "SAS": 2.0},
            {"zinc_id": "z2", "smiles": "CCN", "logP": 1.2, "qed": 0.4, "SAS": 2.1},
        ]
    )
    csv_path = tmp_path / "zinc_small.csv"
    frame.to_csv(csv_path, index=False)

    manifest = build_zinc_graph_corpus(tmp_path, csv_path)
    manifest_path = tmp_path / "graph_corpus_cache" / "zinc_small" / "graph_corpus" / "manifest.pkl"
    legacy_manifest = {key: value for key, value in manifest.items() if key != "bucket_files"}
    with open(manifest_path, "wb") as handle:
        pickle.dump(legacy_manifest, handle)

    graphs, metadata = load_zinc_graph_dataset(tmp_path, max_molecules=10)

    assert len(graphs) == 2
    assert metadata["zinc_id"].tolist() == ["z1", "z2"]

    with open(manifest_path, "rb") as handle:
        repaired_manifest = pickle.load(handle)
    assert "bucket_files" in repaired_manifest
    assert repaired_manifest["bucket_files"][3].endswith("graphs_nodes_003.pkl")


def test_load_zinc_graph_dataset_repairs_legacy_bucket_payload(tmp_path):
    frame = pd.DataFrame(
        [
            {"zinc_id": "z1", "smiles": "CCO", "logP": 1.0, "qed": 0.5, "SAS": 2.0},
            {"zinc_id": "z2", "smiles": "CCN", "logP": 1.2, "qed": 0.4, "SAS": 2.1},
        ]
    )
    csv_path = tmp_path / "zinc_small.csv"
    frame.to_csv(csv_path, index=False)

    manifest = build_zinc_graph_corpus(tmp_path, csv_path)
    bucket_path = tmp_path / "graph_corpus_cache" / "zinc_small" / manifest["bucket_files"][3]
    with open(bucket_path, "rb") as handle:
        current_items = pickle.load(handle)

    legacy_items = {
        "graphs": [graph for graph, _row in current_items],
        "metadata": pd.DataFrame([row for _graph, row in current_items]),
        "node_count": 3,
    }
    with open(bucket_path, "wb") as handle:
        pickle.dump(legacy_items, handle)

    graphs, metadata = load_zinc_graph_dataset(tmp_path, max_molecules=10)

    assert len(graphs) == 2
    assert metadata["zinc_id"].tolist() == ["z1", "z2"]

    with open(bucket_path, "rb") as handle:
        repaired_items = pickle.load(handle)
    assert isinstance(repaired_items, list)
    assert len(repaired_items) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in repaired_items)
