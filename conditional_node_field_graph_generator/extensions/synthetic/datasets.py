"""Synthetic dataset builders."""

import random
import networkx as nx
import numpy as np
from toolz import curry

from .primitives import make_graph_generator


def _normalize_hash_value(value):
    if isinstance(value, np.ndarray):
        return tuple(_normalize_hash_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_hash_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _normalize_hash_value(val)) for key, val in value.items()))
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _graph_hash(graph):
    hashed_graph = nx.Graph()
    hashed_graph.add_nodes_from(graph.nodes())
    hashed_graph.add_edges_from(graph.edges())

    for node, attrs in graph.nodes(data=True):
        hashed_graph.nodes[node]["_dedupe_label"] = repr(
            tuple(sorted((key, _normalize_hash_value(val)) for key, val in attrs.items()))
        )
    for u, v, attrs in graph.edges(data=True):
        hashed_graph.edges[u, v]["_dedupe_label"] = repr(
            tuple(sorted((key, _normalize_hash_value(val)) for key, val in attrs.items()))
        )
    return nx.weisfeiler_lehman_graph_hash(
        hashed_graph,
        node_attr="_dedupe_label",
        edge_attr="_dedupe_label",
    )


class _FallbackGraphHashDeduper:
    def __init__(self):
        self._seen_hashes = set()

    def fit_filter(self, graphs):
        self._seen_hashes = set()
        return self.filter(graphs)

    def filter(self, graphs):
        unique_graphs = []
        for graph in graphs:
            graph_key = _graph_hash(graph)
            if graph_key in self._seen_hashes:
                continue
            self._seen_hashes.add(graph_key)
            unique_graphs.append(graph)
        return unique_graphs


def _make_duplicate_detection_estimator():
    try:
        from abstractgraph.hashing import GraphHashDeduper
    except ImportError as exc:
        GraphHashDeduper = _FallbackGraphHashDeduper
    return GraphHashDeduper()


class AttributeGenerator(object):
    def __init__(self, data_mtx, targets):
        self.target_classes = sorted(list(set(targets)))
        self.num_classes = len(self.target_classes)
        self.attributes = [data_mtx[[i for i, y in enumerate(targets) if y == t]] for t in self.target_classes]

    def transform(self, class_seq):
        attribute_list = []
        for y in class_seq:
            attributes = self.attributes[y]
            idx = np.random.randint(len(attributes))
            attribute_list.append(attributes[idx].flatten())
        return attribute_list


@curry
def make_graph(graph_generator, alphabet_size, attribute_generator):
    graph = graph_generator
    nx.set_edge_attributes(graph, "-", "label")

    if attribute_generator is not None:
        num_classes = attribute_generator.num_classes
    else:
        num_classes = alphabet_size

    labels = np.random.randint(num_classes, size=nx.number_of_nodes(graph))
    labels_dict = {node_idx: label for node_idx, label in enumerate(labels)}
    nx.set_node_attributes(graph, labels_dict, "true_label")

    labels_dict = {node_idx: label % alphabet_size for node_idx, label in enumerate(labels)}
    nx.set_node_attributes(graph, labels_dict, "label")

    if attribute_generator is not None:
        attributes = attribute_generator.transform(labels)
        attributes_dict = {node_idx: attribute for node_idx, attribute in enumerate(attributes)}
        nx.set_node_attributes(graph, attributes_dict, "vec")
    return graph.copy()


class ArtificialGraphConstructor(object):
    def __init__(self, graph_type="cycle", instance_size=4, alphabet_size=3, attribute_generator=None):
        self.graph_type = graph_type
        self.instance_size = instance_size
        self.alphabet_size = alphabet_size
        self.attribute_generator = attribute_generator
        self.graph_generator = make_graph_generator(graph_type, instance_size)

    def sample(self, n_samples=1):
        samples = [
            make_graph(self.graph_generator, self.alphabet_size, self.attribute_generator)
            for _ in range(n_samples)
        ]
        if n_samples == 1:
            return samples[0]
        return samples


def _make_alphabet(size, kind="int", offset=0):
    if size <= 0:
        raise ValueError("Alphabet size must be >= 1.")

    if kind == "int":
        return list(range(offset, offset + size))

    if kind == "letter":
        if offset + size > 26:
            raise ValueError("Letter alphabet supports at most 26 symbols across all components.")
        return [chr(ord("A") + i) for i in range(offset, offset + size)]

    raise ValueError("kind must be 'int' or 'letter'.")


def _make_component_alphabets(size, kind="int", component_specific_alphabets=True):
    components = ("cycle", "path", "star")
    if not component_specific_alphabets:
        shared_alphabet = _make_alphabet(size, kind)
        return {component: shared_alphabet for component in components}
    return {
        component: _make_alphabet(size, kind, offset=component_index * size)
        for component_index, component in enumerate(components)
    }


def generate_cycle_path_star_graph(
    cycle_length,
    path_length,
    num_rays,
    ray_length,
    node_alphabet_size=1,
    edge_alphabet_size=1,
    node_alphabet_kind="int",
    edge_alphabet_kind="int",
    component_specific_alphabets=True,
    seed=None,
):
    """Generate one connected cycle -> path -> star-ray NetworkX graph."""

    if cycle_length < 0 or path_length < 0 or num_rays < 0 or ray_length < 0:
        raise ValueError("Structural parameters must be non-negative.")

    rng = random.Random(seed)
    node_labels_by_component = _make_component_alphabets(
        node_alphabet_size,
        node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    edge_labels_by_component = _make_component_alphabets(
        edge_alphabet_size,
        edge_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )

    graph = nx.Graph()
    next_node = 0

    def assign_node_label(node, component):
        graph.nodes[node]["label"] = rng.choice(node_labels_by_component[component])
        graph.nodes[node]["label_component"] = component

    def add_node(role, component):
        nonlocal next_node
        node = next_node
        next_node += 1
        graph.add_node(node, role=role)
        assign_node_label(node, component)
        return node

    def add_edge(u, v, role, component):
        graph.add_edge(
            u,
            v,
            label=rng.choice(edge_labels_by_component[component]),
            role=role,
            label_component=component,
        )

    if cycle_length == 0:
        cycle_nodes = [add_node("cycle_anchor", "cycle")]
    elif cycle_length < 3:
        raise ValueError("cycle_length must be 0 or >= 3 for a simple cycle.")
    else:
        cycle_nodes = [add_node("cycle", "cycle") for _ in range(cycle_length)]
        for i in range(cycle_length):
            add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % cycle_length], "cycle", "cycle")

    anchor = rng.choice(cycle_nodes)
    graph.nodes[anchor]["role"] = "cycle_anchor"

    current = anchor
    for _ in range(path_length):
        new_node = add_node("path", "path")
        add_edge(current, new_node, "path", "path")
        current = new_node

    hub = current
    graph.nodes[hub]["role"] = "star_hub"
    assign_node_label(hub, "star")

    for ray_id in range(num_rays):
        current = hub
        if ray_length == 0:
            leaf = add_node(f"ray_{ray_id}_leaf", "star")
            add_edge(hub, leaf, "star_ray", "star")
            continue

        for step in range(ray_length):
            role = f"ray_{ray_id}_node"
            if step == ray_length - 1:
                role = f"ray_{ray_id}_leaf"
            new_node = add_node(role, "star")
            add_edge(current, new_node, "star_ray", "star")
            current = new_node

    graph.graph["metadata"] = {
        "cycle_length": cycle_length,
        "path_length": path_length,
        "num_rays": num_rays,
        "ray_length": ray_length,
        "node_alphabet_size": node_alphabet_size,
        "edge_alphabet_size": edge_alphabet_size,
        "component_specific_alphabets": component_specific_alphabets,
        "node_alphabets_by_component": node_labels_by_component,
        "edge_alphabets_by_component": edge_labels_by_component,
    }

    assert nx.is_connected(graph)
    return graph


def _sample_int_parameter(value, rng, name, *, minimum=None, valid_values=None):
    """Resolve either a fixed integer or an inclusive integer range."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} range must be a 2-tuple: (min, max).")
        low, high = value
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError(f"{name} range bounds must be integers.")
        if high < low:
            raise ValueError(f"{name} range max must be >= min.")
        candidates = list(range(low, high + 1))
        if minimum is not None:
            candidates = [candidate for candidate in candidates if candidate >= minimum]
        if valid_values is not None:
            candidates = [candidate for candidate in candidates if valid_values(candidate)]
        if not candidates:
            raise ValueError(f"{name} range {value!r} contains no valid values.")
        return rng.choice(candidates)

    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or a 2-tuple integer range.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if valid_values is not None and not valid_values(value):
        raise ValueError(f"{name} has invalid value {value!r}.")
    return value


def generate_artificial_dataset(
    num_graphs,
    cycle_length,
    path_length,
    num_rays,
    ray_length,
    node_alphabet_size=1,
    edge_alphabet_size=1,
    node_alphabet_kind="int",
    edge_alphabet_kind="int",
    component_specific_alphabets=True,
    seed=None,
):
    """Generate cycle -> path -> star-ray artificial NetworkX graphs."""

    if not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a non-negative integer.")

    rng = random.Random(seed)
    graphs = []

    for _ in range(num_graphs):
        sampled_cycle_length = _sample_int_parameter(
            cycle_length,
            rng,
            "cycle_length",
            minimum=0,
            valid_values=lambda candidate: candidate == 0 or candidate >= 3,
        )
        sampled_path_length = _sample_int_parameter(path_length, rng, "path_length", minimum=0)
        sampled_num_rays = _sample_int_parameter(num_rays, rng, "num_rays", minimum=0)
        sampled_ray_length = _sample_int_parameter(ray_length, rng, "ray_length", minimum=0)
        sampled_node_alphabet_size = _sample_int_parameter(
            node_alphabet_size,
            rng,
            "node_alphabet_size",
            minimum=1,
        )
        sampled_edge_alphabet_size = _sample_int_parameter(
            edge_alphabet_size,
            rng,
            "edge_alphabet_size",
            minimum=1,
        )

        graphs.append(
            generate_cycle_path_star_graph(
                cycle_length=sampled_cycle_length,
                path_length=sampled_path_length,
                num_rays=sampled_num_rays,
                ray_length=sampled_ray_length,
                node_alphabet_size=sampled_node_alphabet_size,
                edge_alphabet_size=sampled_edge_alphabet_size,
                node_alphabet_kind=node_alphabet_kind,
                edge_alphabet_kind=edge_alphabet_kind,
                component_specific_alphabets=component_specific_alphabets,
                seed=rng.randint(0, 2**32 - 1),
            )
        )

    return graphs


def link_graphs(graph_source, graph_target, n_link_edges=0):
    n = nx.number_of_nodes(graph_source)
    graph_source_endpoints = np.random.randint(nx.number_of_nodes(graph_source), size=n_link_edges)
    graph_target_endpoints = np.random.randint(nx.number_of_nodes(graph_target), size=n_link_edges)
    graph = nx.disjoint_union(graph_source, graph_target)
    for u, v in graph_source.edges():
        graph.edges[u, v]["true_label"] = "source"
    for u, v in graph_target.edges():
        graph.edges[u + n, v + n]["true_label"] = "destination"
    for s, t in zip(graph_source_endpoints, graph_target_endpoints):
        graph.add_edge(s, t + n, label="-", true_label="joint")
    return graph


def make_graphs(
    graph_generator_target_type,
    graph_generator_context_type,
    target_size,
    context_size,
    alphabet_size,
    attribute_generator,
    n_link_edges,
    num_graphs,
    use_single_target=True,
):
    context_graphs = []
    for _ in range(num_graphs):
        graph_generator = make_graph_generator(graph_generator_context_type, context_size)
        context_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
        context_graphs.append(context_graph.copy())

    if use_single_target:
        graph_generator = make_graph_generator(graph_generator_target_type, target_size)
        target_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
        target_graphs = [target_graph.copy()] * num_graphs
    else:
        target_graphs = []
        for _ in range(num_graphs):
            graph_generator = make_graph_generator(graph_generator_target_type, target_size)
            target_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
            target_graphs.append(target_graph.copy())

    graphs = [
        link_graphs(graph_source=target_graph, graph_target=context_graph, n_link_edges=n_link_edges)
        for target_graph, context_graph in zip(target_graphs, context_graphs)
    ]
    return graphs


def make_graphs_classification_dataset(
    graph_generator_target_type,
    graph_generator_context_type,
    target_size,
    context_size,
    alphabet_size,
    n_link_edges,
    num_graphs,
    attribute_generator=None,
):
    pos_graphs = make_graphs(
        graph_generator_target_type,
        graph_generator_context_type,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    neg_graphs = make_graphs(
        graph_generator_target_type,
        graph_generator_context_type,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=False,
    )
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs


def make_two_types_graphs_classification_dataset(
    graph_generator_target_type_pos,
    graph_generator_context_type_pos,
    graph_generator_target_type_neg,
    graph_generator_context_type_neg,
    target_size,
    context_size,
    alphabet_size,
    n_link_edges,
    num_graphs,
    attribute_generator=None,
):
    pos_graphs = make_graphs(
        graph_generator_target_type_pos,
        graph_generator_context_type_pos,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    neg_graphs = make_graphs(
        graph_generator_target_type_neg,
        graph_generator_context_type_neg,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs


class ArtificialGraphDatasetConstructor(object):
    def __init__(
        self,
        graph_generator_target_type_pos,
        graph_generator_context_type_pos,
        graph_generator_target_type_neg,
        graph_generator_context_type_neg,
        target_size_pos,
        context_size_pos,
        alphabet_size_pos,
        n_link_edges_pos,
        target_size_neg,
        context_size_neg,
        alphabet_size_neg,
        n_link_edges_neg,
        attribute_generator=None,
    ):
        self.graph_generator_target_type_pos = graph_generator_target_type_pos
        self.graph_generator_context_type_pos = graph_generator_context_type_pos
        self.graph_generator_target_type_neg = graph_generator_target_type_neg
        self.graph_generator_context_type_neg = graph_generator_context_type_neg
        self.target_size_pos = target_size_pos
        self.context_size_pos = context_size_pos
        self.alphabet_size_pos = alphabet_size_pos
        self.n_link_edges_pos = n_link_edges_pos
        self.target_size_neg = target_size_neg
        self.context_size_neg = context_size_neg
        self.alphabet_size_neg = alphabet_size_neg
        self.n_link_edges_neg = n_link_edges_neg
        self.attribute_generator = attribute_generator

    def get_graph_types(self):
        return ["path", "tree", "cycle", "degree", "regular", "dense"]

    def sample(self, n_samples, return_separate_classes=False):
        pos_graphs = make_graphs(
            self.graph_generator_target_type_pos,
            self.graph_generator_context_type_pos,
            self.target_size_pos,
            self.context_size_pos,
            self.alphabet_size_pos,
            self.attribute_generator,
            self.n_link_edges_pos,
            n_samples,
            use_single_target=True,
        )
        neg_graphs = make_graphs(
            self.graph_generator_target_type_neg,
            self.graph_generator_context_type_neg,
            self.target_size_neg,
            self.context_size_neg,
            self.alphabet_size_neg,
            self.attribute_generator,
            self.n_link_edges_neg,
            n_samples,
            use_single_target=True,
        )
        gdde = _make_duplicate_detection_estimator()
        pos_graphs = gdde.fit_filter(pos_graphs)
        neg_graphs = gdde.filter(neg_graphs)
        targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
        graphs = pos_graphs + neg_graphs
        if return_separate_classes:
            return pos_graphs, neg_graphs
        return graphs, targets


__all__ = [
    "ArtificialGraphConstructor",
    "ArtificialGraphDatasetConstructor",
    "AttributeGenerator",
    "generate_artificial_dataset",
    "generate_cycle_path_star_graph",
    "link_graphs",
    "make_graph",
    "make_graphs",
    "make_graphs_classification_dataset",
    "make_two_types_graphs_classification_dataset",
]
