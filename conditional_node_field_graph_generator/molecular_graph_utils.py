"""Standalone utilities for molecular graph loading, conversion, drawing, and caching."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        return obj


ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_ZINC_TARGET_COLUMNS = ("logP", "qed", "SAS")
PUBCHEM_FILENAME_TEMPLATE = "AID{assay_id}_{split}.sdf"
logger = logging.getLogger(__name__)


def smiles_to_networkx_molecule(
    smiles: str,
    zinc_id: Optional[str] = None,
    properties: Optional[dict] = None,
) -> Optional[nx.Graph]:
    """Convert a SMILES string into a discrete-labelled molecular graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return molecule_to_networkx(
        mol,
        graph_metadata={
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            **({"zinc_id": str(zinc_id)} if zinc_id is not None else {}),
            **(properties or {}),
        },
    )


def molecule_to_networkx(
    mol: Chem.Mol,
    graph_metadata: Optional[dict] = None,
) -> nx.Graph:
    """Convert an RDKit molecule to a NetworkX graph with discrete labels."""
    graph = nx.Graph()
    if graph_metadata:
        graph.graph.update(graph_metadata)

    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            label=atom.GetSymbol(),
            atomic_num=int(atom.GetAtomicNum()),
            formal_charge=int(atom.GetFormalCharge()),
            aromatic=bool(atom.GetIsAromatic()),
        )

    for bond in mol.GetBonds():
        bond_label = "AROMATIC" if bond.GetIsAromatic() else str(int(round(bond.GetBondTypeAsDouble())))
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=bond_label,
            aromatic=bool(bond.GetIsAromatic()),
        )
    return graph


def rdkmol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """Compatibility wrapper matching the historical CoCoGraPE loader API."""
    return molecule_to_networkx(mol)


def sdf_to_nx(file: str | Path):
    """Yield NetworkX graphs from an SDF file."""
    supplier = Chem.SDMolSupplier(str(file))
    for mol in supplier:
        if mol is not None:
            yield rdkmol_to_nx(mol)


def smi_to_nx(file: str | Path):
    """Yield NetworkX graphs from a SMILES file."""
    supplier = Chem.SmilesMolSupplier(str(file))
    for mol in supplier:
        if mol is not None:
            yield rdkmol_to_nx(mol)


def networkx_to_molecule(
    graph: nx.Graph,
    sanitize: bool = True,
) -> Chem.Mol:
    """Convert a discrete-labelled molecular NetworkX graph back to an RDKit molecule."""
    rw_mol = Chem.RWMol()
    node_to_atom_idx: dict[int, int] = {}
    sorted_nodes = sorted(graph.nodes())

    for node_id in sorted_nodes:
        attrs = graph.nodes[node_id]
        atomic_num = attrs.get("atomic_num")
        label = attrs.get("label", "C")
        atom = Chem.Atom(int(atomic_num) if atomic_num is not None else str(label))
        atom.SetFormalCharge(int(attrs.get("formal_charge", 0)))
        atom.SetIsAromatic(bool(attrs.get("aromatic", False)))
        node_to_atom_idx[node_id] = rw_mol.AddAtom(atom)

    for u, v, attrs in graph.edges(data=True):
        bond_label = attrs.get("label", "1")
        if bond_label == "AROMATIC" or attrs.get("aromatic", False):
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = {
                "1": Chem.BondType.SINGLE,
                "2": Chem.BondType.DOUBLE,
                "3": Chem.BondType.TRIPLE,
            }.get(str(bond_label), Chem.BondType.SINGLE)
        rw_mol.AddBond(node_to_atom_idx[u], node_to_atom_idx[v], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            bond = rw_mol.GetBondBetweenAtoms(node_to_atom_idx[u], node_to_atom_idx[v])
            if bond is not None:
                bond.SetIsAromatic(True)

    mol = rw_mol.GetMol()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("aromatic", False):
            atom = mol.GetAtomWithIdx(node_to_atom_idx[node_id])
            atom.SetIsAromatic(True)
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def nx_to_rdkit(graph: nx.Graph) -> Chem.Mol:
    """Compatibility wrapper matching the historical CoCoGraPE drawing API."""
    return networkx_to_molecule(graph, sanitize=False)


def _prepare_molecule_for_drawing(graph: nx.Graph) -> Chem.Mol:
    mol = nx_to_rdkit(graph)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    try:
        return Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    except Exception:
        return mol


def set_coordinates(compounds: Iterable[Chem.Mol]) -> None:
    """Compute 2D coordinates for RDKit molecules before drawing."""
    for mol in compounds:
        if mol is None:
            raise ValueError("set_coordinates failed on a null molecule.")
        mol.UpdatePropertyCache(strict=False)
        AllChem.Compute2DCoords(mol)


def compounds_to_image(
    compounds: Iterable[Chem.Mol],
    n_graphs_per_line: int = 5,
    size: int = 250,
    legend: Optional[Iterable[str]] = None,
):
    """Compatibility wrapper matching the historical CoCoGraPE drawing API."""
    compound_list = list(compounds)
    if not compound_list:
        return None
    set_coordinates(compound_list)
    return Draw.MolsToGridImage(
        compound_list,
        molsPerRow=max(1, int(n_graphs_per_line)),
        subImgSize=(int(size), int(size)),
        legends=list(legend) if legend is not None else None,
    )


def molecule_graphs_to_grid_image(
    graphs: Iterable[nx.Graph],
    legends: Optional[Iterable[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 200),
):
    """Render molecular graphs to an RDKit grid image."""
    graph_list = list(graphs)
    if not graph_list:
        return None
    prepared = [_prepare_molecule_for_drawing(graph) for graph in graph_list]
    if legends is None:
        legend_list = []
        for graph in graph_list:
            legend_list.append(
                str(
                    graph.graph.get("smiles")
                    or graph.graph.get("zinc_id")
                    or graph.graph.get("pubchem_cid")
                    or ""
                )
            )
    else:
        legend_list = [str(value) for value in legends]
    return Draw.MolsToGridImage(
        prepared,
        legends=legend_list,
        molsPerRow=max(1, int(mols_per_row)),
        subImgSize=sub_img_size,
        useSVG=False,
    )


def nx_to_image(
    graphs: Iterable[nx.Graph],
    n_graphs_per_line: int = 5,
    size: int = 250,
    title_key: Optional[str] = None,
    titles: Optional[Iterable[str]] = None,
):
    """Compatibility wrapper matching the historical CoCoGraPE drawing API."""
    graph_list = list(graphs)
    if title_key:
        legends = [graph.graph.get(title_key, "N/A") for graph in graph_list]
    elif titles is not None:
        legends = list(titles)
    else:
        legends = [str(index) for index in range(len(graph_list))]
    compounds = [_prepare_molecule_for_drawing(graph) for graph in graph_list]
    return compounds_to_image(
        compounds,
        n_graphs_per_line=n_graphs_per_line,
        size=size,
        legend=legends,
    )


def draw_molecules(
    graphs: Iterable[nx.Graph],
    titles: Optional[Iterable[str]] = None,
    num: Optional[int] = None,
    n_graphs_per_line: int = 7,
    size: int = 7,
    legends: Optional[Iterable[str]] = None,
    mols_per_row: Optional[int] = None,
    sub_img_size: Optional[tuple[int, int]] = None,
):
    """Display a grid image for a list of molecular graphs."""
    graph_list = list(graphs)
    if num is not None:
        graph_list = graph_list[:num]
    if not graph_list:
        print("No molecules to display.")
        return None

    if titles is None and legends is None:
        legends = [str(index) for index in range(len(graph_list))]
    elif legends is None and titles is not None:
        legends = list(titles)[: len(graph_list)]

    if mols_per_row is None:
        mols_per_row = n_graphs_per_line
    if sub_img_size is None:
        pixel_size = max(150, int(size) * 35)
        sub_img_size = (pixel_size, pixel_size)

    for start in range(0, len(graph_list), 50):
        image = molecule_graphs_to_grid_image(
            graph_list[start : start + 50],
            legends=None if legends is None else list(legends)[start : start + 50],
            mols_per_row=mols_per_row,
            sub_img_size=sub_img_size,
        )
        if image is not None:
            display(image)
    return None


def resolve_pubchem_dir(pubchem_dir: Path | str | None = None) -> Path:
    """Resolve the directory that stores assay-level PubChem SDF files."""
    if pubchem_dir is not None:
        return Path(pubchem_dir).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "notebooks" / "datasets" / "PUBCHEM"


def _pubchem_sdf_path(pubchem_dir: Path | str, assay_id: str, split: str) -> Path:
    return resolve_pubchem_dir(pubchem_dir) / PUBCHEM_FILENAME_TEMPLATE.format(assay_id=assay_id, split=split)


class SupervisedDataSetLoader(object):
    """Local copy of the historical CoCoGraPE supervised dataset loader."""

    def __init__(
        self,
        load_func=None,
        size=None,
        use_targets_list=None,
        use_equalized=False,
        use_multiclass_to_binary=False,
        use_regression_to_binary=False,
        regression_to_binary_threshold=None,
    ):
        self.load_func = load_func
        self.size = size
        self.use_targets_list = use_targets_list
        self.use_equalized = use_equalized
        self.use_multiclass_to_binary = use_multiclass_to_binary
        self.use_regression_to_binary = use_regression_to_binary
        self.regression_to_binary_threshold = regression_to_binary_threshold

    def resize(self, data, targets, size):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        idxs = np.random.choice(len(targets), size=size, replace=False)
        data = [data[idx] for idx in idxs]
        if data_is_numpy:
            data = np.asarray(data)
        targets = [targets[idx] for idx in idxs]
        if target_is_numpy:
            targets = np.asarray(targets)
        return data, targets

    def equalize(self, data, targets):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        target_values = list(sorted(set(targets)))
        idxs_list = [[idx for idx in range(len(targets)) if targets[idx] == target_value] for target_value in target_values]
        min_size = min(len(idxs) for idxs in idxs_list)
        idxs_list = [np.random.choice(idxs, size=min_size, replace=False) for idxs in idxs_list]
        data = sum([[data[idx] for idx in idxs] for idxs in idxs_list], [])
        if data_is_numpy:
            data = np.asarray(data)
        targets = sum([[targets[idx] for idx in idxs] for idxs in idxs_list], [])
        if target_is_numpy:
            targets = np.asarray(targets)
        return data, targets

    def binarize_multiclass(self, targets):
        target_is_numpy = isinstance(targets, np.ndarray)
        targets = [target % 2 for target in targets]
        if target_is_numpy:
            targets = np.asarray(targets)
        return targets

    def binarize_regression(self, targets):
        target_is_numpy = isinstance(targets, np.ndarray)
        targets = [target < self.regression_to_binary_threshold for target in targets]
        if target_is_numpy:
            targets = np.asarray(targets)
        return targets

    def filter_targets(self, data, targets, targets_list):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        idxs = [idx for idx in range(len(targets)) if targets[idx] in targets_list]
        filtered_data = [data[idx] for idx in idxs]
        filtered_targets = [targets[idx] for idx in idxs]
        if data_is_numpy:
            filtered_data = np.asarray(filtered_data)
        if target_is_numpy:
            filtered_targets = np.asarray(filtered_targets)
        return filtered_data, filtered_targets

    def load(self):
        data, targets = self.load_func()
        if self.use_targets_list is not None:
            data, targets = self.filter_targets(data, targets, targets_list=self.use_targets_list)
        if self.use_multiclass_to_binary:
            targets = self.binarize_multiclass(targets)
        if self.use_regression_to_binary:
            targets = self.binarize_regression(targets)
        if self.use_equalized:
            data, targets = self.equalize(data, targets)
        if self.size is not None and len(targets) > self.size:
            data, targets = self.resize(data, targets, size=self.size)
        return data, targets


class RDKitMolFileLoader(object):
    """Local copy of the historical CoCoGraPE RDKit file loader."""

    def __init__(self, dirname=".", filetype="smi"):
        self.dirname = dirname
        self.filetype = filetype

    def load(self, filename):
        full_fname = os.path.join(self.dirname, filename)
        if self.filetype == "sdf":
            graphs = list(sdf_to_nx(full_fname))
        elif self.filetype == "smi":
            graphs = list(smi_to_nx(full_fname))
        else:
            raise ValueError(f"Unsupported molecular filetype: {self.filetype!r}")
        return graphs


class PubChemLoader(object):
    """Local copy of the historical CoCoGraPE PubChem loader."""

    def __init__(self):
        self.root_uri = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        self.pubchem_dir = "PUBCHEM"

    def get_assay_description(self, assay_id):
        fname = f"AID{assay_id}_info.txt"
        full_fname = os.path.join(self.pubchem_dir, fname)
        if not os.path.isfile(full_fname):
            query = self.root_uri + f"assay/aid/{assay_id}/summary/JSON"
            reply = requests.get(query)
            text = reply.json()["AssaySummaries"]["AssaySummary"][0]["Name"]
            with open(full_fname, "w") as file_handle:
                file_handle.write(text)
        else:
            with open(full_fname, "r") as file_handle:
                text = "".join(file_handle.readlines())
        return text

    def _make_rest_query(self, assay_id, active=True):
        mode = "active" if active else "inactive"
        core = f"assay/aid/{assay_id}/cids/JSON?cids_type={mode}&list_return=listkey"
        return self.root_uri + core

    def _get_compounds(self, fname, active, aid, stepsize=50):
        with open(fname, "w") as file_handle:
            index_start = 0
            reply = requests.get(self._make_rest_query(aid, active=active))
            listkey = reply.json()["IdentifierList"]["ListKey"]
            size = reply.json()["IdentifierList"]["Size"]
            for chunk, index_end in enumerate(range(0, size + stepsize, stepsize)):
                if index_end != 0:
                    repeat = True
                    while repeat:
                        logger.debug(
                            "Chunk %s) Processing compounds %s to %s (%s)",
                            chunk,
                            index_start,
                            index_end - 1,
                            size,
                        )
                        query = self.root_uri + f"compound/listkey/{listkey}/SDF?&listkey_start={index_start}&listkey_count={stepsize}"
                        reply = requests.get(query)
                        if "PUGREST.Timeout" in reply.text:
                            logger.debug("PUGREST TIMEOUT")
                        elif "PUGREST.BadRequest" in reply.text:
                            logger.debug("bad request %s %d %d %d", query, chunk, index_end, size)
                            reply = requests.get(self._make_rest_query(aid, active=active))
                            listkey = reply.json()["IdentifierList"]["ListKey"]
                        elif reply.status_code != 200:
                            logger.debug("UNKNOWN ERROR %s", query)
                            logger.debug(reply.status_code)
                            logger.debug(reply.text)
                            raise RuntimeError("PubChem query failed")
                        else:
                            repeat = False
                            file_handle.write(reply.text)
                index_start = index_end

    def _query_db(self, assay_id, fname=None, active=True, stepsize=50):
        self._get_compounds(fname=fname + ".tmp", active=active, aid=assay_id, stepsize=stepsize)
        os.rename(fname + ".tmp", fname)

    def download(self, assay_id, active=True, stepsize=50):
        if not os.path.exists(self.pubchem_dir):
            os.mkdir(self.pubchem_dir)
        fname = f"AID{assay_id}_{'active' if active else 'inactive'}.sdf"
        full_fname = os.path.join(self.pubchem_dir, fname)
        if not os.path.isfile(full_fname):
            logger.debug("Querying PubChem for AID: %s", assay_id)
            self._query_db(assay_id, fname=full_fname, active=active, stepsize=stepsize)
        else:
            logger.debug("Reading from file: %s", full_fname)
        return full_fname

    def load(self, assay_id, dirname="PUBCHEM", format_type="sdf"):
        self.download(assay_id, active=True, stepsize=50)
        self.download(assay_id, active=False, stepsize=50)
        fname_active = f"AID{assay_id}_active.sdf"
        fname_inactive = f"AID{assay_id}_inactive.sdf"
        if format_type == "sdf":
            pos_graphs = RDKitMolFileLoader(dirname=dirname, filetype="sdf").load(fname_active)
            neg_graphs = RDKitMolFileLoader(dirname=dirname, filetype="sdf").load(fname_inactive)
        elif format_type == "smi":
            pos_graphs = RDKitMolFileLoader(dirname=dirname, filetype="smi").load(fname_active)
            neg_graphs = RDKitMolFileLoader(dirname=dirname, filetype="smi").load(fname_inactive)
        else:
            raise ValueError(f"Unsupported format_type={format_type!r}")
        graphs = pos_graphs + neg_graphs
        targets = [1] * len(pos_graphs) + [0] * len(neg_graphs)
        return graphs, targets


def _pubchem_record_to_graph(
    mol: Chem.Mol,
    assay_id: str,
    activity: int,
) -> nx.Graph:
    properties = {name: mol.GetProp(name) for name in mol.GetPropNames()}
    smiles = None
    if mol.HasProp("PUBCHEM_OPENEYE_CAN_SMILES"):
        smiles = mol.GetProp("PUBCHEM_OPENEYE_CAN_SMILES")
    else:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    graph = molecule_to_networkx(
        mol,
        graph_metadata={
            "smiles": smiles,
            "assay_id": str(assay_id),
            "activity": int(activity),
            "pubchem_cid": properties.get("PUBCHEM_COMPOUND_CID"),
            "inchi": properties.get("PUBCHEM_IUPAC_INCHI"),
        },
    )
    graph.graph["pubchem_properties"] = properties
    return graph


def _load_pubchem_split(
    pubchem_dir: Path | str,
    assay_id: str,
    split: str,
    activity: int,
) -> tuple[list[nx.Graph], list[dict]]:
    sdf_path = _pubchem_sdf_path(pubchem_dir, assay_id=assay_id, split=split)
    if not sdf_path.exists():
        raise FileNotFoundError(f"Missing PubChem SDF file: {sdf_path}")

    graphs: list[nx.Graph] = []
    records: list[dict] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        graph = _pubchem_record_to_graph(mol, assay_id=assay_id, activity=activity)
        graphs.append(graph)
        props = graph.graph.get("pubchem_properties", {})
        records.append(
            {
                "assay_id": str(assay_id),
                "activity": int(activity),
                "pubchem_cid": graph.graph.get("pubchem_cid"),
                "smiles": graph.graph.get("smiles"),
                "node_count": int(graph.number_of_nodes()),
                "edge_count": int(graph.number_of_edges()),
                "inchi": graph.graph.get("inchi"),
                **props,
            }
        )
    return graphs, records


def load_pubchem_graph_dataset(
    pubchem_dir: Path | str | None = None,
    assay_id: str = "651610",
    dataset_size: Optional[int] = None,
    max_node_count: Optional[int] = None,
    use_equalized: bool = False,
    random_state: int = 0,
) -> tuple[list[nx.Graph], np.ndarray, pd.DataFrame]:
    """Load PubChem assay graphs from local active/inactive SDF files."""
    dataset_root = resolve_pubchem_dir(pubchem_dir)
    active_graphs, active_records = _load_pubchem_split(dataset_root, assay_id=assay_id, split="active", activity=1)
    inactive_graphs, inactive_records = _load_pubchem_split(
        dataset_root,
        assay_id=assay_id,
        split="inactive",
        activity=0,
    )

    graphs = active_graphs + inactive_graphs
    metadata = pd.DataFrame.from_records(active_records + inactive_records)
    if max_node_count is not None:
        keep_mask = metadata["node_count"].to_numpy() <= int(max_node_count)
        metadata = metadata.loc[keep_mask].reset_index(drop=True)
        graphs = [graph for graph, keep in zip(graphs, keep_mask) if bool(keep)]

    targets = metadata["activity"].to_numpy(dtype=int)
    if use_equalized and len(metadata):
        positive_indices = np.flatnonzero(targets == 1)
        negative_indices = np.flatnonzero(targets == 0)
        sample_size = min(len(positive_indices), len(negative_indices))
        if sample_size > 0:
            rng = np.random.default_rng(random_state)
            chosen = np.concatenate(
                [
                    rng.choice(positive_indices, size=sample_size, replace=False),
                    rng.choice(negative_indices, size=sample_size, replace=False),
                ]
            )
            chosen = np.sort(chosen)
            metadata = metadata.iloc[chosen].reset_index(drop=True)
            graphs = [graphs[int(index)] for index in chosen]
            targets = metadata["activity"].to_numpy(dtype=int)

    if dataset_size is not None and int(dataset_size) < len(graphs):
        rng = np.random.default_rng(random_state)
        chosen = np.sort(rng.choice(len(graphs), size=int(dataset_size), replace=False))
        metadata = metadata.iloc[chosen].reset_index(drop=True)
        graphs = [graphs[int(index)] for index in chosen]
        targets = metadata["activity"].to_numpy(dtype=int)

    return graphs, targets, metadata


def download_zinc_dataset(
    dataset_dir: Path | str,
    url: str = ZINC_250K_URL,
    filename: str = "zinc_250k.csv",
    chunk_size: int = 1 << 20,
    force: bool = False,
) -> Path:
    """Download the ZINC CSV once and persist it to disk."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_path = dataset_root / filename
    if output_path.exists() and not force:
        print(f"Using cached ZINC CSV: {output_path}")
        return output_path

    print(f"Downloading ZINC dataset from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    print(f"Saved ZINC CSV to: {output_path}")
    return output_path


def _iter_property_columns(frame: pd.DataFrame, reserved: Iterable[str]) -> list[str]:
    reserved_names = set(reserved)
    return [column for column in frame.columns if column not in reserved_names]


def _corpus_root(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir).expanduser().resolve() / "graph_corpus"


def _corpus_manifest_path(dataset_dir: Path | str) -> Path:
    return _corpus_root(dataset_dir) / "manifest.pkl"


def _bucket_filename(node_count: int) -> str:
    return f"graphs_nodes_{int(node_count):03d}.pkl"


def extract_zinc_targets(
    metadata: pd.DataFrame,
    target_columns: Iterable[str] = DEFAULT_ZINC_TARGET_COLUMNS,
) -> pd.DataFrame:
    """Extract the numeric ZINC property columns used as per-molecule targets."""
    columns = list(target_columns)
    missing_columns = [column for column in columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(
            "Metadata is missing required ZINC target columns: "
            f"{missing_columns}. Available columns: {metadata.columns.tolist()}"
        )
    return metadata.loc[:, columns].astype(float).reset_index(drop=True)


def build_zinc_graph_corpus(
    dataset_dir: Path | str,
    csv_path: Path | str,
    force: bool = False,
    smiles_column: str = "smiles",
    id_column: str = "zinc_id",
) -> dict:
    """Build the full ZINC NetworkX corpus once and persist per-node-count buckets."""
    csv_file = Path(csv_path).expanduser().resolve()
    corpus_root = _corpus_root(dataset_dir)
    corpus_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _corpus_manifest_path(dataset_dir)

    if manifest_path.exists() and not force:
        with manifest_path.open("rb") as handle:
            manifest = pickle.load(handle)
        print(f"Loaded cached ZINC graph corpus manifest: {manifest_path}")
        return manifest

    frame = pd.read_csv(csv_file)
    property_columns = _iter_property_columns(frame, reserved=[smiles_column, id_column])

    bucket_graphs: dict[int, list[nx.Graph]] = {}
    bucket_records: dict[int, list[dict]] = {}
    invalid_smiles_count = 0
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        graph = smiles_to_networkx_molecule(
            row_dict[smiles_column],
            zinc_id=row_dict.get(id_column),
            properties={column: row_dict[column] for column in property_columns},
        )
        if graph is None:
            invalid_smiles_count += 1
            continue
        node_count = int(graph.number_of_nodes())
        bucket_graphs.setdefault(node_count, []).append(graph)
        bucket_records.setdefault(node_count, []).append(
            {
                id_column: graph.graph.get("zinc_id"),
                smiles_column: graph.graph.get("smiles"),
                "node_count": node_count,
                "edge_count": int(graph.number_of_edges()),
                **{column: row_dict[column] for column in property_columns},
            }
        )

    metadata_frames: list[pd.DataFrame] = []
    bucket_index: dict[int, dict[str, object]] = {}
    for node_count in sorted(bucket_graphs):
        bucket_path = corpus_root / _bucket_filename(node_count)
        bucket_metadata = pd.DataFrame.from_records(bucket_records[node_count])
        payload = {
            "graphs": bucket_graphs[node_count],
            "metadata": bucket_metadata,
            "node_count": int(node_count),
        }
        with bucket_path.open("wb") as handle:
            pickle.dump(payload, handle)
        metadata_frames.append(bucket_metadata)
        bucket_index[int(node_count)] = {
            "path": bucket_path.name,
            "count": int(len(bucket_graphs[node_count])),
        }

    metadata = (
        pd.concat(metadata_frames, ignore_index=True)
        if metadata_frames
        else pd.DataFrame(columns=[id_column, smiles_column, "node_count", "edge_count", *property_columns])
    )
    manifest = {
        "csv_path": str(csv_file),
        "property_columns": property_columns,
        "bucket_index": bucket_index,
        "node_counts": sorted(bucket_index),
        "total_graphs": int(len(metadata)),
        "invalid_smiles_count": int(invalid_smiles_count),
    }
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle)
    print(f"Built ZINC graph corpus with {manifest['total_graphs']} molecules: {corpus_root}")
    return manifest


def _load_bucket(dataset_dir: Path | str, node_count: int) -> tuple[list[nx.Graph], pd.DataFrame]:
    bucket_path = _corpus_root(dataset_dir) / _bucket_filename(node_count)
    with bucket_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["graphs"], payload["metadata"]


def load_zinc_graph_dataset(
    dataset_dir: Path | str,
    max_molecules: int = 100_000,
    min_node_count: Optional[int] = None,
    max_node_count: Optional[int] = 40,
    refresh_download: bool = False,
    refresh_cache: bool = False,
    url: str = ZINC_250K_URL,
) -> tuple[list[nx.Graph], pd.DataFrame]:
    """Load a node-count slice from the cached full ZINC NetworkX corpus."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    csv_path = download_zinc_dataset(
        dataset_root,
        url=url,
        force=refresh_download,
    )
    manifest = build_zinc_graph_corpus(
        dataset_dir=dataset_root,
        csv_path=csv_path,
        force=refresh_cache,
    )

    selected_counts = [
        int(node_count)
        for node_count in manifest["node_counts"]
        if (min_node_count is None or int(node_count) >= int(min_node_count))
        and (max_node_count is None or int(node_count) <= int(max_node_count))
    ]

    graphs: list[nx.Graph] = []
    metadata_frames: list[pd.DataFrame] = []
    remaining = int(max_molecules) if max_molecules is not None else None
    for node_count in selected_counts:
        bucket_graphs, bucket_metadata = _load_bucket(dataset_root, node_count)
        if remaining is None:
            take = len(bucket_graphs)
        else:
            take = min(int(remaining), len(bucket_graphs))
        if take <= 0:
            break
        graphs.extend(bucket_graphs[:take])
        metadata_frames.append(bucket_metadata.iloc[:take].reset_index(drop=True))
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break

    metadata = (
        pd.concat(metadata_frames, ignore_index=True)
        if metadata_frames
        else pd.DataFrame(columns=["zinc_id", "smiles", "node_count", "edge_count"])
    )
    print(
        "Loaded ZINC graph slice "
        f"(n={len(graphs)}, min_node_count={min_node_count}, max_node_count={max_node_count}) "
        f"from cached corpus: {_corpus_root(dataset_root)}"
    )
    return graphs, metadata
