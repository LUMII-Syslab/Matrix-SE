import random
from abc import abstractmethod, ABC

import networkx as nx
import numpy as np


class Graph(ABC):

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
    def input_classes(self):
        pass

    @property
    @abstractmethod
    def output_classes(self):
        pass

    def generator_fn(self, feature_shape, label_shape, training=False) -> callable:
        def _generator():
            while True:
                nodes = np.random.randint(feature_shape[0] // 2, feature_shape[0])
                feature_graph = self.generate_graph(node_count=nodes)

                if len(feature_graph.edges) == 0:
                    continue

                feature_graph = self.set_weights(feature_graph)

                # pos = nx.spring_layout(feature_graph, scale=3)
                # nx.draw_networkx(feature_graph, pos)
                # labels = nx.get_edge_attributes(feature_graph, 'weight')
                # nx.draw_networkx_edge_labels(feature_graph, pos, labels)
                # plt.show()

                feature_graph, label_graph = self.label_fn(feature_graph)

                # pos = nx.spring_layout(label_graph, scale=3)
                # nx.draw_networkx(label_graph, pos)
                # labels = nx.get_edge_attributes(label_graph, 'weight')
                # nx.draw_networkx_edge_labels(label_graph, pos, labels)
                # plt.show()

                yield feature_graph.to_directed(), label_graph.to_directed()

        return _generator


class TriangleFinding(Graph):

    def __init__(self, dense=True) -> None:
        super().__init__()
        self.generate_dense = dense

    @property
    def input_classes(self):
        return 1

    @property
    def output_classes(self):
        return 2

    def set_weights(self, feature_graph):
        return feature_graph

    def label_fn(self, feature: nx.Graph):

        label = feature.copy()
        nx.set_edge_attributes(feature, 0, "feature")

        for e in label.edges:
            v1_neighbor_set = set(label.neighbors(e[0]))
            for v2 in label.neighbors(e[1]):
                if v2 in v1_neighbor_set:
                    label[e[0]][e[1]]['label'] = 1
                else:
                    label[e[0]][e[1]]['label'] = 0

        return feature, label

    def generate_graph(self, node_count):
        if self.generate_dense and node_count > 2:
            # generate a dense bipartite graph
            # there are no triangles in such graph
            n = np.random.randint(1, node_count)
            m = node_count - n
            g_0 = nx.bipartite.random_graph(n, m, 0.5)
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


class ComponentColouring(Graph):

    @property
    def input_classes(self):
        return 100

    @property
    def output_classes(self):
        return 100

    def set_weights(self, feature_graph):
        for (u, v, w) in feature_graph.edges(data=True):
            w['feature'] = np.random.randint(2, self.input_classes)

        return feature_graph

    def label_fn(self, feature: nx.Graph):
        components = [feature.subgraph(c) for c in nx.connected_components(feature)]
        for component in components:
            min_w = self.find_min_weight(component)
            for u, v, w in component.edges(data=True):
                w['label'] = min_w

        return feature, nx.compose_all(components)

    def find_min_weight(self, component: nx.Graph):
        minimal = self.input_classes
        for u, v, w in component.edges(data=True):
            if w['feature'] < minimal:
                minimal = w['feature']

        return minimal

    def generate_graph(self, node_count):
        tree = nx.generators.random_tree(node_count)  # type: nx.Graph
        edges = [edge for edge in tree.edges]
        if len(edges) > 2:
            throw_out = random.randrange(int(np.sqrt(len(edges))) + 2)

            for _ in range(throw_out):
                edge = random.choice(edges)
                edges.remove(edge)
                tree.remove_edge(*edge)
        components = []
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


class Transitivity(Graph):

    @property
    def input_classes(self):
        return 2

    @property
    def output_classes(self):
        return 3

    def set_weights(self, feature_graph):
        return feature_graph

    def label_fn(self, feature: nx.Graph):

        node_count = feature.number_of_nodes()
        complete = nx.complete_graph(node_count)
        nx.set_edge_attributes(complete, 0, "label")  # Non-edges

        attributes = nx.get_edge_attributes(feature, "feature")  # Set normal edges
        nx.set_edge_attributes(complete, attributes, "label")

        pth = dict(nx.all_pairs_shortest_path_length(feature, 2))
        for node1 in feature.nodes:
            for node2 in feature.nodes:
                if node1 in pth and node2 in pth[node1] and pth[node1][node2] == 2:
                    complete[node1][node2]['label'] = 2

        return feature, complete

    def generate_graph(self, node_count):
        eps = 0.02
        p = ((1 + eps) * np.log(node_count)) / node_count
        graph = nx.generators.gnp_random_graph(node_count, p, directed=True)
        nx.set_edge_attributes(graph, 1, "feature")
        return graph
