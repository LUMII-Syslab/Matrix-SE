import tensorflow as tf
from tensorflow.keras.layers import Embedding

from layers.shuffle import LinearTransform
from models.base import Model
from utils.shuffle import gelu


class ConvNet(Model):
    """
    Convolutional Neural Network used for algorithmic tasks (as described in paper)
    """

    def __init__(self, feature_maps) -> None:
        self.feature_maps = feature_maps

        self.embedding = None

        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None

        self.conv_4 = None
        self.conv_5 = None
        self.conv_6 = None

        self.conv_7 = None
        self.conv_8 = None
        self.conv_9 = None

        self.output_linear = None

    @property
    def config(self):
        return self.__config

    def build(self, input_classes, output_classes):
        self.embedding = Embedding(input_classes, output_dim=self.feature_maps)
        self.conv_1 = tf.keras.layers.Conv2D(self.feature_maps, 3, padding="same", activation=gelu)
        self.conv_2 = tf.keras.layers.Conv2D(self.feature_maps, 3, padding="same", activation=gelu)
        self.conv_3 = tf.keras.layers.Conv2D(self.feature_maps, 3, padding="same", activation=gelu)

        self.conv_4 = tf.keras.layers.Conv2D(self.feature_maps * 2, 3, padding="same", activation=gelu)
        self.conv_5 = tf.keras.layers.Conv2D(self.feature_maps * 2, 3, padding="same", activation=gelu)
        self.conv_6 = tf.keras.layers.Conv2D(self.feature_maps * 2, 3, padding="same", activation=gelu)

        self.conv_7 = tf.keras.layers.Conv2D(self.feature_maps * 4, 3, padding="same", activation=gelu)
        self.conv_8 = tf.keras.layers.Conv2D(self.feature_maps * 4, 3, padding="same", activation=gelu)
        self.conv_9 = tf.keras.layers.Conv2D(self.feature_maps * 4, 3, padding="same", activation=gelu)

        self.output_linear = LinearTransform("output", output_classes)

    def call(self, inputs, training=False):
        inputs = self.embedding(inputs)

        block_1 = self.conv_1(inputs)
        block_1 = self.conv_2(block_1)
        block_1 = self.conv_3(block_1)

        block_2 = self.conv_4(block_1)
        block_2 = self.conv_5(block_2)
        block_2 = self.conv_6(block_2)

        block_3 = self.conv_7(block_2)
        block_3 = self.conv_8(block_3)
        block3 = self.conv_9(block_3)

        return self.output_linear(block3)
