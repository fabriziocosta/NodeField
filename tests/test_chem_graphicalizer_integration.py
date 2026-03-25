from abstractgraph_graphicalizer.chem import normalize_graph_schema, smiles_to_graph

from conditional_node_field_graph_generator.extensions.demo.visualization import show_molecules


def test_show_molecules_uses_graphicalizer_drawer(monkeypatch):
    graph = smiles_to_graph("CCO")
    calls = {}

    def fake_draw_molecules(graphs, **kwargs):
        calls["graphs"] = list(graphs)
        calls["kwargs"] = kwargs
        return "figure"

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.visualization.draw_molecules",
        fake_draw_molecules,
    )

    result = show_molecules([graph], legends=["ethanol"])

    assert result is None
    assert calls["graphs"] == [graph]
    assert calls["kwargs"]["titles"] == ["ethanol"]
    assert calls["kwargs"]["n_graphs_per_line"] == 7


def test_show_molecules_can_return_figure_when_requested(monkeypatch):
    graph = smiles_to_graph("CCO")

    monkeypatch.setattr(
        "conditional_node_field_graph_generator.extensions.demo.visualization.draw_molecules",
        lambda *args, **kwargs: "figure",
    )

    result = show_molecules([graph], return_figure=True)

    assert result == "figure"


def test_normalize_graph_schema_upgrades_legacy_numeric_bond_labels():
    graph = smiles_to_graph("CC")
    for _, _, data in graph.edges(data=True):
        data["label"] = "1"
        data.pop("bond_order", None)
        data.pop("bond_type", None)

    normalized = normalize_graph_schema(graph)

    assert next(iter(normalized.edges(data=True)))[2]["label"] == "single"
