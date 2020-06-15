import random
from abc import abstractmethod

import networkx as nx
import numpy as np
import tensorflow as tf
from networkx.algorithms import bipartite

import utils.data as data_utils
from data.base import GeneratorDataset


class Graph(GeneratorDataset):

    @abstractmethod
    def label_fn(self, feature):
        pass

    @abstractmethod
    def generate_graph(self, node_count):
        pass

    @abstractmethod
    def set_weights(self, feature_graph):
        pass

    @property
    @abstractmethod
    def non_edge(self):
        pass

    def __init__(self) -> None:
        super().__init__({
            "input_classes": 3,
            "output_classes": 3,
            "train_lengths": [8, 16, 32],
            "eval_lengths": [8, 16, 32]
        })

    def generator_fn(self, feature_shape, label_shape, training=False) -> callable:
        def _generator():
            while True:
                nodes = np.random.randint(feature_shape[0] // 2, feature_shape[0])
                feature_graph = self.generate_graph(node_count=nodes)
                feature_graph = self.set_weights(feature_graph)

                feature = nx.to_numpy_matrix(feature_graph, dtype=np.int32, nonedge=self.non_edge)

                label_graph = self.label_fn(feature_graph)
                label = nx.to_numpy_matrix(label_graph, dtype=np.int32, nonedge=self.non_edge)

                feature = data_utils.pad_with_zeros(feature, feature_shape)
                label = data_utils.pad_with_zeros(label, label_shape)

                yield feature, label

        return _generator

    @property
    def generator_output_types(self):
        return tf.int32, tf.int32

    @property
    def train_output_shapes(self) -> list:
        return [((x, x), (x, x)) for x in self.config["train_lengths"]]

    @property
    def eval_output_shapes(self) -> list:
        return [((x, x), (x, x)) for x in self.config["eval_lengths"]]

    @property
    def train_size(self) -> int:
        return 1000000

    @property
    def eval_size(self) -> int:
        return 20000


class Transitivity(Graph):

    @property
    def non_edge(self):
        return 2

    def set_weights(self, feature_graph):
        return feature_graph

    def label_fn(self, feature: nx.Graph):
        label = feature.copy()

        pth = dict(nx.all_pairs_shortest_path_length(feature, 2))  # Find transitive paths of length 2
        for node1 in feature.nodes:
            for node2 in feature.nodes:
                if node1 in pth and node2 in pth[node1] and pth[node1][node2] == 2:
                    label.add_edge(node1, node2)

        return label

    def generate_graph(self, node_count):
        eps = 0.02
        p = ((1 + eps) * np.log(node_count)) / node_count  # Calculate probability
        return nx.generators.gnp_random_graph(node_count, p, directed=True)


class TriangleFinding(Graph):
    def __init__(self, dense=True) -> None:
        super().__init__()
        self.generate_dense = dense

    @property
    def non_edge(self):
        return 2

    def set_weights(self, feature_graph):
        return feature_graph

    def label_fn(self, feature: nx.Graph):
        label = nx.Graph()
        label.add_nodes_from(feature.nodes)
        for e in feature.edges:
            v1_neighbor_set = set(feature.neighbors(e[0]))
            for v2 in feature.neighbors(e[1]):
                if v2 in v1_neighbor_set:
                    label.add_edge(e[0], e[1])
                    break

        return label

    def generate_graph(self, node_count):
        if self.generate_dense and node_count > 2:
            # generate a dense bipartite graph
            # there are no triangles in such graph
            n = np.random.randint(1, node_count)
            m = node_count - n
            g_0 = bipartite.random_graph(n, m, 0.5)
            # randomly shuffle the nodes.
            g = nx.Graph()
            shuffled_nodes = list(g_0.nodes)
            g.add_nodes_from(shuffled_nodes)
            random.shuffle(shuffled_nodes)
            for e in g_0.edges:
                g.add_edge(shuffled_nodes[e[0]], shuffled_nodes[e[1]])

            # add a few edges which may form triangles
            n_edges = int(np.log(node_count)) + 1
            nodes = list(g.nodes)
            for _ in range(n_edges):
                v1 = random.choice(nodes)
                v2 = random.choice(nodes)
                if v1 != v2:
                    g.add_edge(v1, v2)
            return g
        else:
            # generate a random graph with such sparsity that several triangles are expected.
            eps = 0.2
            p = ((1 + eps) * np.log(node_count)) / node_count
            return nx.generators.gnp_random_graph(node_count, p, directed=False)


class ComponentLabeling(Graph):

    def __init__(self) -> None:
        super().__init__()
        self.add_config("input_classes", 100)
        self.add_config("output_classes", 100)

    @property
    def non_edge(self):
        return 1

    def set_weights(self, feature_graph):
        for (u, v, w) in feature_graph.edges(data=True):
            w['weight'] = np.random.randint(2, self.config["input_classes"])

        return feature_graph

    def label_fn(self, feature: nx.Graph):
        components = [feature.subgraph(c) for c in nx.connected_components(feature)]
        for component in components:
            min_w = self.find_min_weight(component)
            for u, v, w in component.edges(data=True):
                w['weight'] = min_w

        return nx.compose_all(components)

    def find_min_weight(self, component: nx.Graph):
        minimal = self.config['input_classes']
        for u, v, w in component.edges(data=True):
            if w['weight'] < minimal:
                minimal = w['weight']

        return minimal

    def generate_graph(self, node_count):
        # generate random trees
        tree = nx.generators.random_tree(node_count)  # type: nx.Graph
        edges = [edge for edge in tree.edges]
        if len(edges) > 2:
            # throw out random edges of tree forming connected components
            throw_out = random.randrange(int(np.sqrt(len(edges))) + 2)

            for _ in range(throw_out):
                edge = random.choice(edges)
                edges.remove(edge)
                tree.remove_edge(*edge)
        components = []

        # add additional random edges to each component
        for component_nodes in nx.connected_components(tree):  # type: nx.Graph
            component = tree.subgraph(component_nodes)
            component = nx.Graph(component)
            non_edges = [x for x in nx.non_edges(component)]

            if len(non_edges) > 1:
                add_edges = np.log2(len(non_edges))
                add_edges = random.randrange(int(add_edges) + 1)

                for _ in range(add_edges):
                    edge = random.choice(non_edges)
                    non_edges.remove(edge)
                    component.add_edge(*edge)

            components.append(component)
        feature_graph = nx.compose_all(components)
        return feature_graph
