from pathlib import Path

import numpy as np
import pandas as pd

from conditional_node_field_graph_generator.extensions.molecular import (
    PubChemLoader,
    SupervisedDataSetLoader,
    build_zinc_graph_corpus,
    extract_zinc_targets,
    nx_to_rdkit,
    smiles_to_networkx_molecule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBCHEM_DATA_ROOT = PROJECT_ROOT / "notebooks" / "datasets" / "PUBCHEM"


def test_smiles_round_trip_to_rdkit():
    graph = smiles_to_networkx_molecule("CCO", zinc_id="z1", properties={"qed": 0.5})

    assert graph is not None
    assert graph.graph["zinc_id"] == "z1"
    assert graph.graph["qed"] == 0.5
    assert graph.number_of_nodes() == 3

    mol = nx_to_rdkit(graph)
    assert mol.GetNumAtoms() == 3


def test_local_pubchem_loader_reads_bundled_sdf_files():
    loader = PubChemLoader()
    loader.pubchem_dir = str(PUBCHEM_DATA_ROOT)

    graphs, targets = loader.load("651610", dirname=str(PUBCHEM_DATA_ROOT))

    assert len(graphs) == len(targets)
    assert len(graphs) > 100
    assert set(np.unique(np.asarray(targets))).issubset({0, 1})


def test_supervised_dataset_loader_equalizes_and_resizes():
    loader = PubChemLoader()
    loader.pubchem_dir = str(PUBCHEM_DATA_ROOT)

    def load_func():
        return loader.load("651610", dirname=str(PUBCHEM_DATA_ROOT))

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
