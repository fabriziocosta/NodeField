"""Maintained NodeField modules."""

__all__ = [
    "ConditionalNodeFieldGraphDecoder",
    "ConditionalNodeFieldGraphGenerator",
    "ConditionalNodeFieldGenerator",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
