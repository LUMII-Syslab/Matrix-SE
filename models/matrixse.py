import tensorflow as tf
from tensorflow.keras.layers import Embedding

from layers.quaternary_shuffle import BenesBlock
from layers.shuffle import LinearTransform
from models.base import Model


class MatrixSE(Model):
    """
    Model for 2D algorithmic tasks (graphs and matrices).
    Matrix Shuffle-Exchange is generalization of Neural Shuffle-Exchange
    (https://papers.nips.cc/paper/8889-neural-shuffle-exchange-networks-sequence-processing-in-on-log-n-time)
    network to two dimensions.
    """

    def __init__(self, feature_maps, block_count) -> None:
        self.__config = {
            "num_units": feature_maps,
            "block_count": block_count
        }

        self.benes_block = None
        self.embedding = None
        self.output_layer = None

    @property
    def config(self):
        return self.__config

    def build(self, input_classes, output_classes):
        self.embedding = Embedding(input_classes, output_dim=self.config["num_units"],
                                   embeddings_initializer=tf.truncated_normal_initializer(stddev=0.25))

        self.benes_block = BenesBlock(self.config["block_count"], self.config["num_units"])

        self.output_layer = LinearTransform("output", output_classes)

    def call(self, inputs, training=False):
        embedding = self.embedding(inputs)

        benes_block = self.benes_block(embedding, training=training)

        return self.output_layer(benes_block)
