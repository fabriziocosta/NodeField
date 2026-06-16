"""Maintained NodeField modules."""

__all__ = [
    "ConditionalNodeFieldGraphDecoder",
    "ConditionalNodeFieldGraphGenerator",
    "ConditionalNodeFieldGenerator",
    "feasibility_effort_map",
    "resolve_feasibility_effort",
]


def __getattr__(name: str):
    if name == "ConditionalNodeFieldGenerator":
        from .conditional_node_field_generator import ConditionalNodeFieldGenerator

        return ConditionalNodeFieldGenerator
    if name == "ConditionalNodeFieldGraphDecoder":
        from .conditional_node_field_graph_decoder import ConditionalNodeFieldGraphDecoder

        return ConditionalNodeFieldGraphDecoder
    if name == "ConditionalNodeFieldGraphGenerator":
        from .conditional_node_field_graph_generator import ConditionalNodeFieldGraphGenerator

        return ConditionalNodeFieldGraphGenerator
    if name == "feasibility_effort_map":
        from .feasibility_effort import feasibility_effort_map

        return feasibility_effort_map
    if name == "resolve_feasibility_effort":
        from .feasibility_effort import resolve_feasibility_effort

        return resolve_feasibility_effort
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
