# Graph Generator Architecture

This document explains the architecture of `ConditionalNodeFieldGraphGenerator`, the component that turns graphs into training supervision, coordinates the conditional node generator, and reconstructs final `networkx` graphs through the decoder.

Implementation anchors:

- [`../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py)
- [`../conditional_node_field_graph_generator/graph_generator_state.py`](../conditional_node_field_graph_generator/graph_generator_state.py)
- [`../conditional_node_field_graph_generator/interpolation_utils.py`](../conditional_node_field_graph_generator/interpolation_utils.py)
- [`../conditional_node_field_graph_generator/oracle_utils.py`](../conditional_node_field_graph_generator/oracle_utils.py)
- [`../conditional_node_field_graph_generator/feasibility_utils.py`](../conditional_node_field_graph_generator/feasibility_utils.py)
- [`../conditional_node_field_graph_generator/persistence.py`](../conditional_node_field_graph_generator/persistence.py)
- [`../conditional_node_field_graph_generator/conditional_node_field_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_generator.py)
- [`3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md)
- [`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md)
- [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)
- [`2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md`](2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md)
- [`2D_TARGET_GUIDANCE_README.md`](2D_TARGET_GUIDANCE_README.md)
- [`4_MAIN_CLASS_INTERFACES_README.md`](4_MAIN_CLASS_INTERFACES_README.md)

## Scope

The graph generator is the orchestration layer above the Conditional Node Field model.

It is responsible for:

1. fitting graph-level and node-level vectorizers,
2. extracting supervision targets from training graphs,
3. deciding which semantic channels are learned, constant, or disabled,
4. assembling the `NodeGenerationBatch` consumed by the node generator,
5. invoking the conditional node generator at training and inference time,
6. handing generated node-level predictions to the graph decoder,
7. optionally filtering decoded graphs with a feasibility estimator.

It is not itself the neural model and it is not itself the combinatorial decoder. It is the layer that binds those parts into one end-to-end graph pipeline.

## Current Module Split

The maintained implementation is no longer just one monolithic orchestration file.

- `conditional_node_field_graph_generator.py`
  owns the public `ConditionalNodeFieldGraphGenerator` class and the end-to-end orchestration flow.

- `graph_generator_state.py`
  groups long-lived generator configuration and mutable streamed-fit counters into dedicated dataclasses.

- `interpolation_utils.py`
  owns graph-conditioning interpolation primitives such as magnitude-aware spherical interpolation and integer count interpolation.

- `oracle_utils.py`
  owns oracle-specific helper types and pure functions such as violating-node-set normalization and temporary edge-memory penalties.

- `feasibility_utils.py`
  owns retry-loop formatting helpers used only for feasibility-attempt reporting.

- `persistence.py`
  owns full-generator save/load behavior, schema checks, and saved-generator name resolution.

The class is still large, but the support logic is now split into smaller modules with clearer ownership.

## Main Components

At a high level, the architecture has four collaborating parts.

```mermaid
flowchart LR
    G[Training / Input Graphs]

    subgraph REP[Representation]
        GV[Graph Vectorizer]
        NV[Node Vectorizer]
    end

    subgraph BATCH[Batch Objects]
        GCB[GraphConditioningBatch]
        NGB[NodeGenerationBatch]
        GNB[GeneratedNodeBatch]
    end

    subgraph MODEL[Generation]
        NG[Conditional Node Field Generator]
    end

    subgraph DECODE[Reconstruction]
        DEC[Conditional Node Field Graph Decoder + ILP]
        NX[Decoded Graphs]
    end

    G --> GV --> GCB
    G --> NV --> NGB
    GCB --> NG
    NGB --> NG
    NG --> GNB
    GCB --> DEC
    GNB --> DEC --> NX

    classDef data fill:#f6efe5,stroke:#9a6b2f,stroke-width:1.2px,color:#2f2419;
    classDef model fill:#e7f0ea,stroke:#2e6a4f,stroke-width:1.2px,color:#173728;
    classDef decode fill:#e8eef7,stroke:#3d5f8c,stroke-width:1.2px,color:#1d2d44;
    classDef group fill:#fbfbfb,stroke:#b8b8b8,stroke-width:1px,color:#333;

    class G,GCB,NGB,GNB data;
    class NG model;
    class DEC,NX decode;
    class REP,BATCH,MODEL,DECODE group;
```

### 1. Graph-Level Vectorizer

`graph_vectorizer` maps a full graph to a fixed-width graph embedding used as part of the conditioning signal.

This embedding represents global graph context and latent semantics. It is combined with explicit graph statistics:

- node count,
- edge count.

Those three pieces form `GraphConditioningBatch`.

### 2. Node-Level Vectorizer

`node_graph_vectorizer` maps each graph to a variable-length matrix of node embeddings.

Those matrices are the training targets for the conditional node generator. During inference, the node generator tries to reconstruct node-level representations that are compatible with the requested conditioning vector.

### 3. Conditional Node Generator

`conditional_node_generator_model` is usually `ConditionalNodeFieldGenerator`.

It receives:

- graph-level conditioning,
- padded node-level training examples,
- semantic supervision such as node degrees, node labels, edge existence, and optional auxiliary locality.

It predicts:

- node existence,
- node degrees,
- optional node labels,
- optional edge probabilities,
- optional edge labels.

The internal Conditional Node Field mechanics are described in [`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md), the training/loss details are described in [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md), and the hyperparameter/metrics interpretation is described in [`2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md`](2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md).

### 4. Graph Decoder

`graph_decoder` converts generated node-level outputs into final graph objects.

Its most important job is structural reconstruction:

- turn soft node existence and degree predictions into a valid node set,
- turn edge probabilities into a binary adjacency matrix,
- enforce structural consistency with a solver,
- attach node and edge labels according to the supervision plan.

The decoder and solver details are documented in [`3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md).

## Data Model

The graph generator works with a few explicit batch abstractions.

### `GraphConditioningBatch`

This holds graph-level conditions:

- `graph_embeddings`
- `node_counts`
- `edge_counts`

This object is used both during training and generation.

`node_counts` is a global size condition, not a per-slot support decision. It tells the generator how large the graph should be, but it does not determine which latent node slots will survive into the final graph.

### `NodeGenerationBatch`

This holds padded node-level supervision for the conditional model:

- `node_embeddings_list`
- `node_presence_mask`
- `node_degree_targets`
- optional `node_label_targets`
- optional direct edge supervision
- optional edge label supervision
- optional auxiliary locality supervision

`node_presence_mask` is not redundant with `node_counts`.

It serves two roles:

- during training, it marks which padded rows correspond to real nodes,
- during generation, it represents the local occupancy pattern that realizes the global node-count target.

This distinction matters because generation is not treated as “the first `k` rows exist by construction”. The model is allowed to explore several candidate node slots and then gradually coalesce onto a final support set.

### `GeneratedNodeBatch`

This is the output of the conditional node generator during inference:

- node presence mask,
- predicted degrees,
- optional predicted node labels,
- optional edge probability matrices,
- optional edge label matrices,
- optional node-label logits and probabilities,
- optional edge-existence probabilities,
- optional edge-label logits and probabilities.

The decoder consumes this object and reconstructs `networkx` graphs.
The richer probability tensors keep full decoder shapes and are intended for
inspection, analysis, and future oracle logic. The current decoder still relies
on the hard channels (`node_labels`, `edge_probability_matrices`,
`edge_label_matrices`) for reconstruction.

Conceptually:

- `node_counts` says how many nodes the graph should contain,
- `node_presence_mask` says which specific node slots are currently materialized.

The current architecture keeps both because the generation process is allowed to be gradual and competitive rather than committing to a fixed support immediately.

## Training Architecture

The `fit()` path in `ConditionalNodeFieldGraphGenerator` follows a clear sequence.

```mermaid
flowchart TD
    A[Training Graphs] --> B[Fit Vectorizers]
    A --> C[Inspect Labels]
    C --> D[Build Supervision Plan]
    B --> E[Raw Graph Context]
    B --> F[Raw Node Embeddings]
    E --> S[Optional TruncatedSVD Compression]
    F --> S
    S --> H[GraphConditioningBatch]
    S --> I[NodeGenerationBatch]
    A --> G[Build Structural Targets]
    D --> G
    G --> I
    H --> J[Conditional Node Field Setup]
    I --> J
    J --> K[Conditional Node Field Fit]
    K --> L[Fitted Graph Generator]

    classDef data fill:#f6efe5,stroke:#9a6b2f,stroke-width:1.2px,color:#2f2419;
    classDef process fill:#f7f4ea,stroke:#8a7a3d,stroke-width:1.2px,color:#3a3218;
    classDef model fill:#e7f0ea,stroke:#2e6a4f,stroke-width:1.2px,color:#173728;

    class A,H,I,L data;
    class B,C,D,E,F,G,S process;
    class J,K model;
```

## Streaming Fit Architecture

`fit_from_stream(...)` uses a schema-frozen warmup and then trains from replayable streamed batches.

`batch_size` keeps the same meaning as in the regular in-memory training path: it is the training batch size. In the streamed path, it is simply the number of compatible streamed graphs grouped into each optimization batch, and it also sets the size of the reserved validation batch.

The flow is:

1. read `warmup_size` graphs from the selected source,
2. fit vectorizers, supervision metadata, feasibility estimator, and model schema on warmup only,
3. train on all warmup graphs during epoch 1,
4. reserve the first compatible post-warmup batch as a fixed validation subset,
5. continue training on the remaining post-warmup stream, skipping incompatible graphs,
6. when `maximum_epochs > 1`, restart only the post-warmup stream for later epochs.

This means warmup plays three roles at once:

- schema definition,
- initial training data for epoch 1.

Validation is intentionally disjoint from warmup training. The fixed validation batch comes from the stream immediately after warmup, not from the warmup graphs themselves.

The schema is never expanded after warmup. Post-warmup graphs are accepted only if they remain compatible with the warmup-defined node-count limit, label vocabularies, and transform/supervision assumptions.

With float-valued `limit`, replayed epochs may naturally expose different Bernoulli-sampled tails when no fixed `random_state` is supplied. With a fixed `random_state`, replay is deterministic.

### Step 1. Fit External Encoders

The generator first fits:

- `graph_vectorizer`
- `node_graph_vectorizer`

If a feasibility estimator is present, it is also fitted here.

This means the graph generator is responsible for learning both the graph-level condition space and the node-level target space before neural training begins.

### Step 2. Inspect Graph Labels

The graph generator extracts:

- per-node label targets via `graphs_to_node_label_targets()`
- per-edge label targets via `graphs_to_edge_label_targets()`

Important current behavior:

- if no node labels exist anywhere, a constant dummy node label is inserted,
- if node labels are mixed between present and missing, `fit()` fails fast,
- if usable edge labels are absent, the edge-label channel is disabled.

This inspection stage prevents the node generator and decoder from training heads that are impossible or pointless for a given dataset.

### Step 3. Build A Supervision Plan

The supervision plan is one of the most important architectural ideas in the codebase.

`_build_supervision_plan()` decides, per channel, whether it is:

- `learned`
- `constant`
- `disabled`

Channels currently include:

- node labels,
- edge labels,
- direct edges,
- auxiliary locality.

Examples:

- if all node labels are the same, node labels become `constant` rather than learned,
- if no usable edge labels are present, edge labels become `disabled`,
- direct edge supervision is enabled by default because the decoder needs structural edge scores,
- auxiliary locality is enabled only when `locality_horizon > 1`.

This plan is then attached to both the node generator and the decoder, so they interpret the same dataset semantics consistently.

### Step 4. Encode Graphs

`encode()` combines:

- `node_encode()` from the node-level vectorizer,
- `graph_encode()` from the graph-level vectorizer.

This creates:

- node embedding targets for training,
- graph conditioning vectors for the conditional model.

When `use_embedding_svd=True`, this encoding stage includes orchestrator-level
`TruncatedSVD` compression before tensors reach the neural model. The generator
fits one SVD on stacked raw node histogram rows and a separate SVD on raw graph
embeddings. `node_encode()`, `graph_encode()`, and `encode()` then return the
compressed arrays for fitted generators. The model trains, samples, and decodes
entirely in this compressed space; there is no inverse transform or attempt to
reconstruct the original sparse histograms.

The graph and node projections are intentionally independent. Graph conditioning
still comes from `graph_vectorizer.transform(graphs)` followed by its own SVD,
not from summing compressed node vectors. If the requested compressed dimension
is greater than or equal to the raw feature width, that side skips SVD and keeps
the raw embeddings.

### Step 5. Build Structural Supervision

If direct edge supervision is enabled, the decoder is used during training-time preprocessing to compute locality supervision:

- horizon-1 direct edge labels,
- optional higher-horizon auxiliary locality labels.

This is an important design choice: the decoder is not only an inference component. It also defines the edge/locality supervision used to train the node generator.

### Step 6. Assemble `NodeGenerationBatch`

`_build_node_batch()` converts the raw graphs and embeddings into explicit supervision tensors:

- padded node masks,
- integer degree targets,
- optional node labels,
- optional edge supervision channels.

The node generator does not discover these targets implicitly. The graph generator constructs them explicitly.

### Step 7. Train The Node Generator

Finally, the graph generator calls:

1. `conditional_node_generator_model.setup(...)`
2. `conditional_node_generator_model.fit(...)`

This separation lets the node generator:

- fit scalers and vocabularies,
- configure supervision heads,
- then launch Lightning training.

When training completes, the graph generator marks itself as fitted and becomes eligible for `decode()`, `sample()`, `conditional_sample()`, `interpolate()`, and `mean()`.

## Inference Architecture

The inference path reuses the same components but in reverse.

```mermaid
flowchart TD
    A[Condition Source]
    B[GraphConditioningBatch]
    C[Conditional Node Field Predict]
    D[GeneratedNodeBatch]
    E[Conditional Node Field Graph Decoder]
    F[Adjacency Solve]
    G[Label Assignment]
    H[Decoded Graphs]
    I[Feasibility Filter]
    J[Accepted Outputs]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    B --> E

    classDef data fill:#f6efe5,stroke:#9a6b2f,stroke-width:1.2px,color:#2f2419;
    classDef model fill:#e7f0ea,stroke:#2e6a4f,stroke-width:1.2px,color:#173728;
    classDef decode fill:#e8eef7,stroke:#3d5f8c,stroke-width:1.2px,color:#1d2d44;

    class A,B,D,H,J data;
    class C model;
    class E,F,G,I decode;
```

### 1. Produce Conditioning

Generation can start from:

- new sampled graph-level conditions via `sample()`,
- conditioning vectors encoded from existing graphs via `decode()` or `conditional_sample()`,
- interpolated conditions via `interpolate()`,
- a mean latent condition via `mean()`.

The conditioning always stays in the `GraphConditioningBatch` format.

### 2. Predict Node-Level Structure And Semantics

`_decode_conditioning_batch()` calls the conditional node generator’s `predict(...)`.

This produces `GeneratedNodeBatch`, which may contain:

- node presence predictions,
- degree predictions,
- node labels,
- edge probabilities,
- edge labels,
- node-label logits and probabilities,
- edge-existence probabilities,
- edge-label logits and probabilities.

The exact set depends on the supervision plan and the fitted model heads.

The intended interpretation is that node existence is an occupancy process, not just a padding artifact. During generation, the model can temporarily distribute probability mass across several candidate node slots. As the relaxation proceeds, those candidates can compete and collapse to a final materialized node set that is consistent with the requested graph size and the rest of the predicted structure.

### 3. Decode Graphs

The graph decoder reconstructs graph objects from `GeneratedNodeBatch`.

This stage:

- decides which nodes exist,
- solves for graph structure,
- assigns labels according to learned/constant/disabled channel modes.

The graph generator itself does not build edges directly. It delegates that responsibility to the decoder.

### 4. Optional Feasibility Filtering And Oracle-Guided Decode

If a feasibility estimator is configured, `_decode_with_feasibility_slots()` can:

- generate multiple candidates per requested graph,
- score them as feasible or infeasible,
- fill output slots only with accepted graphs,
- retry missing slots for several rounds.

This adds a second layer of quality control after decoding.

Architecturally, this is separate from the ILP decode:

- the decoder enforces local structural constraints,
- the feasibility estimator acts as a post-hoc accept/reject model for domain-specific validity.

There is now a second integration path as well:

- when `use_feasibility_oracle=True` and the estimator exposes
  `violating_edge_sets(graphs)`,
- the generator uses the estimator during structural decode itself,
- returned violating edge sets are converted into no-good cuts and the adjacency
  ILP is re-solved for a bounded number of rounds,
- optional node-label and edge-label repairs are treated as soft follow-up
  proposals after each structural candidate rather than as independent decode
  stages,
- those relabeling proposals are accepted only when the full oracle state
  improves,
- within one graph's oracle trace, edges that repeatedly appear inside violating
  sets also accumulate a temporary soft penalty before the next ILP solve.

For the maintained demo pipeline, the feasibility estimator is now exposed as a
stack of internal checks with a boolean activation mask. That lets experiments
run one-hot configurations such as `[1, 0, 0, 0]`, cumulative configurations
such as `[1, 1, 0, 0]`, or the full stack `[1, 1, 1, 1]` without changing the
generator API.

The same demo estimator also supports per-level structural-cut budgets during
oracle-guided decode. By default, these budgets use an adaptive policy:

- assign a decreasing prior budget to earlier and cheaper estimator levels,
- observe how many violating edge sets each level returned for the current
  decoded graph,
- keep each level up to its prior allocation,
- redistribute leftover budget to levels that still have additional violations.

So the default behavior is no longer a rigid fixed ratio of cuts per estimator.
It is a graph-local adaptive redistribution rule around a decreasing prior.

So the same feasibility estimator can contribute in two ways:

- as a separation oracle during adjacency reconstruction,
- as a post-hoc rejection filter after decode.

That temporary edge memory is deliberately local to one graph decode. It is not
shared across graphs, batches, or future calls. Hard cuts still forbid exact
previously observed violating edge sets, while the soft memory simply lowers the
logit of edges that keep participating in bad motifs during the current trace.

## Control Flow By Public API

### `fit(graphs, train_node_generator=True, targets=None)`

Full training orchestration:

1. fit vectorizers,
2. inspect labels,
3. build supervision plan,
4. encode graphs,
5. build supervision tensors,
6. train the node generator.

### `decode(graph_conditioning, ...)`

Decode a supplied `GraphConditioningBatch` directly into graphs.

Use this when conditioning vectors are already available.

By default, decode now attempts oracle-guided feasibility cuts before falling
back to the usual final feasibility filtering stage. If the configured
feasibility estimator does not expose `violating_edge_sets(...)`, decode
silently reuses the previous behavior.

Saved generators loaded through the maintained persistence helper are also
upgraded on load when possible so older demo feasibility-estimator composites
pick up the masked interface and the default adaptive oracle-cut policy without
requiring notebook changes.

When `max_feasibility_seconds_per_sample` is configured, timeout-protected
feasibility filtering is applied per requested output rather than across the
whole batch. In that mode, each slot has its own retry budget and optional
fallback path, and the generator emits one final aggregate generation summary at
the end of the call.

When resolving a saved generator, the persistence helper also accepts the
original unsanitized model name used at save time. This matters for names that
contain characters such as `.` which are normalized in persisted filenames.

### `sample(n_samples, ...)`

Sample graph-level conditions from cached training conditioning, then decode them.

By default, this samples stored graph-conditioning rows directly. When `interpolate_between_n_samples` is provided, each requested output first draws a small subset of cached training conditioning rows, scores candidate pairs by cosine similarity on the cached graph-vectorizer embeddings, samples a pair, and linearly interpolates graph embedding, node count, and edge count to form a new conditioning vector.

When feasibility filtering is active, the final log line reports:

- `requested`
- `returned`
- `feasible`
- `unfiltered`
- `rejected`

along with fractions relative to `requested`.

### `conditional_sample(graphs, n_samples, ...)`

Encode each input graph, repeat each condition `n_samples` times, then decode multiple generated variants per input graph.

Like `decode(...)` and `sample(...)`, this path now enables the feasibility
oracle by default and only bypasses it when disabled per call or unsupported by
the configured estimator.

### `score_feasible_rate(n_samples, max_feasibility_attempts, feasibility_candidates_per_attempt, ...)`

Sample graph-level conditions, decode under feasibility filtering, and return a score dictionary whose main objective is:

- `score`
  Equal to candidate-level `feasible_rate`.

This is intended for hyperparameter search when you want to measure how often the generator-decoder pipeline produces feasible outputs under a fixed retry budget.

The returned dictionary also includes:

- `feasible_rate`
- `fulfilled_rate`
- `accepted_slots`
- `generated_candidates`
- `feasible_candidates`

### `interpolate(G1, G2, k, interpolation_mode)`

Encode two graphs, interpolate in graph-conditioning space, interpolate node/edge counts separately, then decode each intermediate point.

### `mean(graphs)`

Compute a conditioning barycentre from several graphs and decode a single representative graph.

## Architectural Strengths

The current design has a few strong properties.

### Explicit Separation Between Representation And Reconstruction

The node generator learns a continuous conditional representation problem.

The decoder handles discrete structural consistency afterward.

This is cleaner than forcing the neural model to output a valid graph directly.

### Dataset-Adaptive Supervision

The supervision plan prevents unnecessary heads from being trained.

That keeps behavior sensible for datasets with:

- no node labels,
- constant node labels,
- no usable edge labels,
- optional auxiliary locality.

### Reusable Conditioning Space

The graph-level conditioning abstraction supports:

- direct reconstruction,
- random generation,
- interpolation,
- mean-graph synthesis,
- conditional guidance through both supported routes:
- classifier-free guidance (CFG) via target-conditioning channels,
- separate post-hoc guidance via an auxiliary classifier or regressor.

The same decode machinery is reused across all of them.

The two supported guidance modes are:

- CFG through the ordinary `decode(...)`, `sample(...)`, and `conditional_sample(...)` path with
  `desired_target` plus `guidance_scale`
- separate post-hoc guidance through the predictor-specific classifier-guided and regression-guided methods

They are documented in detail in [`2D_TARGET_GUIDANCE_README.md`](2D_TARGET_GUIDANCE_README.md).

### Explicit Failure Modes

The graph generator now fails fast when:

- it is used before `fit()`,
- node labels are inconsistently present,
- feasibility filtering cannot fill outputs and failure mode is `raise`,
- the decoder reports solver failure.

For streamed fitting, a `None` value for `stream_batch_timeout_seconds` no
longer disables protection for the fit call. The streamed path now uses a
temporary safe default timeout so stalled post-warmup batch preparation is
skipped instead of blocking indefinitely, while warmup batch preparation itself
remains unbounded.

That is operationally much safer than silent partial failure.

## Architectural Risks And Limitations

The main tradeoffs are also clear.

### Tight Coupling Between Orchestrator And Training Semantics

`ConditionalNodeFieldGraphGenerator` still knows a lot about:

- label semantics,
- locality supervision policy,
- decoder-assisted supervision,
- feasibility filtering,
- how the node generator should be configured.

That makes it powerful, but also means it is not yet a thin orchestration layer.
That remains true even after extracting helper, state, and persistence support
into smaller modules.

### Structural Quality Depends On Two Models

Good graph reconstruction depends on both:

- the node generator producing good edge and degree signals,
- the decoder successfully reconciling them.

If either side is weak, output quality suffers.

### Scaling Pressure In The Decoder

The graph generator architecture assumes decode-time optimization is acceptable for the graph sizes of interest.

For larger graphs, the solver can become the bottleneck even if the neural generator is fast.

### Vectorizer Quality Is Critical

The system inherits the inductive biases of:

- `graph_vectorizer`
- `node_graph_vectorizer`

If those embeddings are weak or unstable, the downstream Conditional Node Field model and decoder cannot fully compensate.

## Extension Points

The current architecture is designed to allow targeted replacement of major pieces.

### Swap The Graph-Level Vectorizer

You can replace `graph_vectorizer` as long as it exposes a compatible `fit()` / `transform()` interface and produces fixed-width graph embeddings.

### Swap The Node-Level Vectorizer

You can replace `node_graph_vectorizer` as long as it produces per-graph node embedding matrices in a stable node order.

Both vectorizers may return sparse matrices. If orchestrator SVD compression is
enabled, sparse node rows are stacked with `scipy.sparse.vstack` and compressed
with `sklearn.decomposition.TruncatedSVD` before neural training.

### Swap The Conditional Node Generator

Any model implementing the `ConditionalNodeGeneratorBase` interface can replace the Conditional Node Field model if it supports:

- `setup(...)`
- `fit(...)`
- `predict(...)`

### Swap The Decoder

The graph decoder can be replaced independently if it can consume `GeneratedNodeBatch`-style outputs and reconstruct graphs with the same supervision-plan semantics.

### Add Domain-Specific Validity Checks

A feasibility estimator can be attached without changing the neural model or the decoder formulation.

This is useful when “valid” means more than just satisfying degree/connectivity constraints.

## Recommended Reading Order

For someone new to the codebase, the fastest way to build accurate context is:

1. [`../README.md`](../README.md)
2. this file
3. [`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md)
4. [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)
5. [`2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md`](2C_CONDITIONAL_NODE_FIELD_OPTIMIZATION_README.md)
6. [`3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md)
7. [`../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py)
8. [`../conditional_node_field_graph_generator/conditional_node_field_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_generator.py)
