# TODO

This file tracks the work that is still genuinely open after the cleanup passes.

## Model / Feature Work

- Add full tokenized-conditioning support at the graph-generator level so [`ConditionalNodeFieldGraphGenerator`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/conditional_node_field_graph_generator.py) can work directly with structured condition memories, not only flat graph-level conditioning vectors plus node and edge count channels.

  This should cover:
  - tokenized conditioning emitted directly by graph encoders
  - graph-to-graph conditioning based on node embeddings from a previous graph
  - abstract-graph conditioning where high-level motif or scaffold tokens drive concrete graph generation
