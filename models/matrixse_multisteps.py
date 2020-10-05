import tensorflow as tf
from tensorflow.keras.layers import Embedding

from layers.RSE_network import gelu
from layers.quaternary_shuffle import BenesBlock
from layers.shuffle import LinearTransform
from models.base import Model


class MatrixSEMultistep(Model):
    """
    Model for 2D logical reasoning tasks, e.g., Sudoku.
    Model for training Matrix Shuffle-Exchange same as Reccurent Relation Network (https://arxiv.org/abs/1711.08028).
    """
    def __init__(self, feature_maps, block_count, train_steps, eval_steps) -> None:
        self.__config = {
            "num_units": feature_maps,
            "block_count": block_count
        }
        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.embedding = None
        self.benes_block = None
        self.gelu_layer = None
        self.output_layer = None

    @property
    def config(self):
        return self.__config

    def build(self, input_classes, output_classes):
        self.embedding = Embedding(input_classes, output_dim=self.config["num_units"],
                                   embeddings_initializer=tf.truncated_normal_initializer(stddev=0.25))

        self.benes_block = BenesBlock(self.config["block_count"], self.config["num_units"])
        self.gelu_layer = LinearTransform("relu_layer", self.config["num_units"])

        self.output_layer = LinearTransform("output", output_classes)

    def call(self, inputs, training=False):
        embedding = self.embedding(inputs)
        steps = self.train_steps if training else self.eval_steps

        last_layer = embedding
        results = []

        for _ in range(steps):
            last_layer = self.benes_block(last_layer)
            output = self.gelu_layer(last_layer)
            output = gelu(output)
            output = self.output_layer(output)
            results.append(output)

        return results
