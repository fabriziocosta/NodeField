import re

import networkx as nx

from conditional_node_field_graph_generator.extensions.synthetic import (
    generate_artificial_dataset,
    generate_cycle_path_star_graph,
)


def _graphs_equal(graphs_a, graphs_b):
    if len(graphs_a) != len(graphs_b):
        return False
    return all(
        nx.is_isomorphic(
            graph_a,
            graph_b,
            node_match=lambda attrs_a, attrs_b: attrs_a == attrs_b,
            edge_match=lambda attrs_a, attrs_b: attrs_a == attrs_b,
        )
        and graph_a.graph == graph_b.graph
        for graph_a, graph_b in zip(graphs_a, graphs_b)
    )


def test_generate_artificial_dataset_saves_config_and_prints_filename(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)

    generate_artificial_dataset(
        num_graphs=1,
        cycle_length=(3, 4),
        path_length=2,
        num_rays=2,
        ray_length=(1, 2),
        node_alphabet_size=(1, 2),
        edge_alphabet_size=1,
        seed=7,
    )

    output = capsys.readouterr().out
    match = re.search(r"Saved artificial dataset config: (.+\.yaml)", output)

    assert match is not None
    assert (tmp_path / match.group(1)).exists()
    assert match.group(1) == "artificial-cycle-path-star-n1-c3-4-p2-r2x1-2-na1-2.yaml"


def test_generate_artificial_dataset_loads_saved_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    expected_graphs = generate_artificial_dataset(
        num_graphs=3,
        cycle_length=(3, 5),
        path_length=(0, 2),
        num_rays=2,
        ray_length=(1, 2),
        node_alphabet_size=(1, 2),
        edge_alphabet_size=(1, 2),
        seed=11,
    )
    config_path = tmp_path / "artificial-cycle-path-star-n3-c3-5-p0-2-r2x1-2-na1-2-ea1-2.yaml"
    assert config_path.exists()

    loaded_graphs = generate_artificial_dataset(load_from_file=config_path, save_config=False)

    assert _graphs_equal(loaded_graphs, expected_graphs)
    assert len(list(tmp_path.glob("artificial-cycle-path-star-*.yaml"))) == 1


def test_generate_cycle_path_star_graph_can_chain_edge_sharing_cycles():
    graph = generate_cycle_path_star_graph(
        cycle_length=4,
        num_cycles=3,
        path_length=1,
        num_rays=0,
        ray_length=0,
        seed=5,
    )

    assert graph.graph["metadata"]["num_cycles"] == 3
    assert graph.number_of_nodes() == 4 + (3 - 1) * (4 - 2) + 1
    assert graph.number_of_edges() == 4 + (3 - 1) * (4 - 1) + 1
    assert nx.is_connected(graph)
    assert len(nx.cycle_basis(graph)) == 3


def test_generate_artificial_dataset_saves_and_loads_num_cycles_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    expected_graphs = generate_artificial_dataset(
        num_graphs=2,
        cycle_length=4,
        num_cycles=3,
        path_length=1,
        num_rays=0,
        ray_length=0,
        seed=17,
    )
    config_path = tmp_path / "artificial-cycle-path-star-n2-c4-nc3-p1-r0x0.yaml"
    assert config_path.exists()

    loaded_graphs = generate_artificial_dataset(load_from_file=config_path, save_config=False)

    assert _graphs_equal(loaded_graphs, expected_graphs)
    assert all(graph.graph["metadata"]["num_cycles"] == 3 for graph in loaded_graphs)
