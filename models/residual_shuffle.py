import tensorflow as tf
from tensorflow.keras.layers import Embedding

from layers.RSE_network import ResidualShuffleExchange2D
from layers.shuffle import ConvLinear
from models.base import Model


class ResidualShuffle(Model):

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

        self.benes_block = ResidualShuffleExchange2D(self.config["block_count"], self.config["num_units"])

        self.output_layer = ConvLinear("output", 1, output_classes)

    def call(self, inputs, training=False):
        embedding = self.embedding(inputs)

        benes_block = self.benes_block(embedding, training=training)

        return self.output_layer(benes_block)
