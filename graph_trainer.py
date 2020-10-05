import itertools
import time

import dgl
import torch
import torch.nn.functional as F
from prefetch_generator import background
from torch import nn

from graph_networks.datasets import Graph, Transitivity, TriangleFinding
from graph_networks.gin import GIN, MLP
from graph_networks.radam import RAdam


def calculate_accuracy(labels, logits):
    indices = torch.argmax(logits, dim=-1)
    correct = torch.sum(indices == labels)
    return correct * 1.0 / labels.shape[0]


class MLPPredictor(nn.Module):
    def __init__(self, layer_count, in_features, out_classes):
        super().__init__()
        self.mlp = MLP(layer_count, in_features * 2, in_features * 4, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.mlp(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN above.
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, input_classes, out_classes):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_classes, in_features)
        self.model = GIN(5, 2, in_features, hidden_features, out_features, 0.1, False, 'sum', 'sum')
        self.mlp = MLPPredictor(layer_count=3, in_features=out_features, out_classes=out_classes)

    def forward(self, features, labels):
        edge_feat = self.emb(features.edata['feature'])
        h = self.model(features, features.ndata['feature'], edge_feat)
        return self.mlp(labels, h)


@background(max_prefetch=20)
def create_generator(dataset: Graph, node_feature, batch_size, sizes: list):
    generators = [dataset.generator_fn([n, n], [n, n])() for n in sizes]

    while True:
        data = []
        for gen in generators:
            graph_list = [prepare_graph(f, g, node_feature) for (f, g) in itertools.islice(gen, batch_size)]
            feature_graphs, label_graphs = list(zip(*graph_list))
            feature_graphs = dgl.batch(feature_graphs)
            label_graphs = dgl.batch(label_graphs)
            data.append((feature_graphs, label_graphs))

        yield data


def prepare_graph(feature_graph, label_graph, node_feature: torch.Tensor):
    features = dgl.from_networkx(feature_graph, edge_attrs=["feature"], device="cuda:0")
    features.ndata['feature'] = node_feature.repeat([features.num_nodes(), 1])
    labels = dgl.from_networkx(label_graph, edge_attrs=["label"], device="cuda:0")
    return dgl.add_self_loop(features), dgl.add_self_loop(labels)


def main():
    feature_maps = 48 * 7
    steps = 10000
    batch_size = 32
    learning_rate = 0.0001

    dataset = TriangleFinding(dense=True)
    # dataset = ComponentColouring()
    # dataset = Transitivity()

    node_feature = torch.normal(0, 0.25, [1, feature_maps], generator=torch.cuda.manual_seed(1), dtype=torch.float32,
                                device="cuda:0")

    model = Model(in_features=feature_maps,
                  hidden_features=feature_maps,
                  out_features=feature_maps,
                  input_classes=dataset.input_classes,
                  out_classes=dataset.output_classes)
    model = model.cuda()

    opt = RAdam(model.parameters(), lr=learning_rate)
    log_model_parameters(model)
    model = model.train()

    examples = create_generator(dataset=dataset,
                                node_feature=node_feature,
                                batch_size=batch_size,
                                sizes=[8, 16, 32])

    # with torch.autograd.detect_anomaly():
    for step, train_data in enumerate(itertools.islice(examples, steps)):
        start_time = time.time()

        # clear gradients from last step
        opt.zero_grad()

        loss = torch.zeros(1, device="cuda:0")
        accuracy_sum = torch.zeros(1, device="cuda:0")

        for features, labels in train_data:
            # forward propagation by using all nodes
            logits = model(features, labels)

            # # compute loss
            loss += F.cross_entropy(logits, labels.edata["label"])
            accuracy_sum += calculate_accuracy(labels.edata["label"], logits)

        # Do backpropagation
        loss.backward()
        opt.step()

        accuracy = (accuracy_sum / len(train_data)).item()

        line = f"Step: {step}"
        line += f"\tLoss: {loss.item():.6f}"
        line += f"\tAccuracy: {accuracy:.6f}"
        line += f"\tTime elapsed:{time.time() - start_time:.4f}s "

        print(line)

    evaluate_model(dataset, node_feature, model)


def evaluate_model(dataset, node_feature, model):
    model = model.eval()
    size = 8

    while size <= 1024:
        generator = create_generator(dataset=dataset,
                                     node_feature=node_feature,
                                     batch_size=1,
                                     sizes=[size])
        accuracy_sum = 0
        steps = 400

        calculate_time = []

        for examples in itertools.islice(generator, steps):
            features, labels = examples[0]
            with torch.no_grad():
                start_time = time.time()
                logits = model(features, labels)
                calculate_time.append(time.time() - start_time)
                accuracy = calculate_accuracy(labels.edata["label"], logits)
                accuracy_sum += accuracy

        print(f"Eval accuracy on {size} vertices: {accuracy_sum / steps}")

        size *= 2


def log_model_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params}")


if __name__ == '__main__':
    main()
