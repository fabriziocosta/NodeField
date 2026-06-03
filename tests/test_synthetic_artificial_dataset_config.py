import re

import networkx as nx

from conditional_node_field_graph_generator.extensions.synthetic import (
    artificial_node_label_colors,
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


def test_generate_artificial_dataset_returns_graphs_and_plot_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    graphs, plot_artificial_graphs = generate_artificial_dataset(
        num_graphs=1,
        cycle_length=4,
        path_length=1,
        num_rays=1,
        ray_length=1,
        node_alphabet_size=3,
        edge_alphabet_size=1,
        seed=7,
        save_config=False,
    )

    assert isinstance(graphs, list)
    assert callable(plot_artificial_graphs)
    assert plot_artificial_graphs.node_label_colors == {
        0: "#fee2e2",
        1: "#fca5a5",
        2: "#dc2626",
        3: "#dbeafe",
        4: "#93c5fd",
        5: "#2563eb",
        6: "#dcfce7",
        7: "#86efac",
        8: "#16a34a",
    }
    assert "node_label_colors" not in plot_artificial_graphs.plot_kwargs


def test_artificial_node_label_colors_scales_with_alphabet_size():
    colors = artificial_node_label_colors(2)

    assert colors == {
        0: "#fee2e2",
        1: "#dc2626",
        2: "#dbeafe",
        3: "#2563eb",
        4: "#dcfce7",
        5: "#16a34a",
    }


def test_generate_artificial_dataset_loads_saved_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    expected_graphs, _plot_artificial_graphs = generate_artificial_dataset(
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

    loaded_graphs, _loaded_plot_artificial_graphs = generate_artificial_dataset(
        load_from_file=config_path,
        save_config=False,
    )

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

    expected_graphs, _plot_artificial_graphs = generate_artificial_dataset(
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

    loaded_graphs, _loaded_plot_artificial_graphs = generate_artificial_dataset(
        load_from_file=config_path,
        save_config=False,
    )

    assert _graphs_equal(loaded_graphs, expected_graphs)
    assert all(graph.graph["metadata"]["num_cycles"] == 3 for graph in loaded_graphs)
