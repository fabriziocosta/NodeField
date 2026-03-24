from conditional_node_field_graph_generator.extensions.molecular import _impl as molecular_impl
from conditional_node_field_graph_generator.extensions.molecular import smiles_to_networkx_molecule
from conditional_node_field_graph_generator.extensions.molecular import visualization as molecular_visualization


def test_draw_molecules_delegates_to_abstractgraph_renderer(monkeypatch):
    graph = smiles_to_networkx_molecule("CCO")
    calls = {}

    def fake_draw_molecules(graphs, **kwargs):
        calls["graphs"] = list(graphs)
        calls["kwargs"] = kwargs
        return "figure"

    monkeypatch.setattr(molecular_impl, "abstractgraph_draw_molecules", fake_draw_molecules)

    result = molecular_visualization.draw_molecules(
        [graph],
        legends=["ethanol"],
        n_graphs_per_line=3,
        size=5,
    )

    assert result == "figure"
    assert calls["graphs"] == [graph]
    assert calls["kwargs"]["n_graphs_per_line"] == 3
    assert calls["kwargs"]["titles"] == ["ethanol"]
    assert calls["kwargs"]["size"] == (5.0, 3.0)
    assert calls["kwargs"]["show"] is True
