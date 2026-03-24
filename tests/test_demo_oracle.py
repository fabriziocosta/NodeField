import conditional_node_field_graph_generator.conditional_node_field_graph_generator as cngg_module
from conditional_node_field_graph_generator.extensions.demo.oracle import (
    collect_oracle_trace_rows,
    oracle_trace_frame,
    parse_oracle_trace_title,
)


def test_parse_oracle_trace_title_extracts_structured_metrics():
    parsed = parse_oracle_trace_title(
        "Oracle structural graph=0 iteration=2 violating_node_sets=1 violating_edge_sets=3 "
        "new_structural_cuts=4 accepted_structural_cuts=2 log_total=-1.5 "
        "best_log_total=-1.2 best_feasible_log_total=-1.1"
    )

    assert parsed is not None
    assert parsed["phase"] == "structural"
    assert parsed["iteration"] == 2
    assert parsed["violating_node_sets"] == 1
    assert parsed["violating_edge_sets"] == 3
    assert parsed["accepted_structural_cuts"] == 2
    assert parsed["log_total"] == -1.5


def test_collect_oracle_trace_rows_temporarily_wraps_decoder_diagnostics(monkeypatch):
    captured = []

    def fake_original_plot(**kwargs):
        captured.append(kwargs["title"])
        return "ok"

    monkeypatch.setattr(cngg_module, "_plot_decoder_diagnostics", fake_original_plot)

    with collect_oracle_trace_rows() as rows:
        cngg_module._plot_decoder_diagnostics(
            title="Oracle feasibility graph=1 iteration=0 violating_node_sets=0 violating_edge_sets=2 "
            "new_structural_cuts=2 accepted_structural_cuts=1 log_total=-0.5"
        )

    assert len(rows) == 1
    assert rows[0]["phase"] == "feasibility"
    assert captured == [
        "Oracle feasibility graph=1 iteration=0 violating_node_sets=0 violating_edge_sets=2 "
        "new_structural_cuts=2 accepted_structural_cuts=1 log_total=-0.5"
    ]
    assert cngg_module._plot_decoder_diagnostics is fake_original_plot


def test_oracle_trace_frame_returns_dataframe_with_expected_columns():
    frame = oracle_trace_frame(
        [{"phase": "joint", "iteration": 1, "violating_node_sets": 0, "violating_edge_sets": 1}]
    )

    assert frame["phase"].tolist() == ["joint"]
    assert frame["iteration"].tolist() == [1]
