from tensorflow.keras.layers import Conv2D

import utils.shuffle as shuffle_utils
from layers.shuffle import LinearTransform, LayerNormalization, BenesBlock
from models.base import Model


class Switchblade4CIFAR(Model):
    """
    Switchblade with prepended 3 convolutional layers used for CIFAR10
    image classification task.
    """

    def __init__(self, feature_maps, block_count) -> None:
        self.num_units = feature_maps
        self.block_count = block_count

        self.conv = None
        self.conv2 = None
        self.conv3 = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None

        self.input_linear = None
        self.benes_block = None

        self.output_linear = None

    def build(self, input_classes, output_classes):
        self.conv = Conv2D(self.num_units // 2, 2, padding="same", use_bias=False)
        self.norm1 = LayerNormalization(axis=[1, 2])

        self.conv2 = Conv2D(self.num_units, 2, padding="same", use_bias=False)
        self.norm2 = LayerNormalization(axis=[1, 2])

        self.conv3 = Conv2D(self.num_units * 2, 2, padding="same", use_bias=False)
        self.norm3 = LayerNormalization(axis=[1, 2])

        self.input_linear = LinearTransform("input", self.num_units)
        self.benes_block = BenesBlock(self.block_count, self.num_units)
        self.output_linear = LinearTransform("output", output_classes)

    def call(self, inputs, training=False):
        conv1 = self.conv(inputs)
        conv1 = self.norm1(conv1)
        conv1 = shuffle_utils.gelu(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.norm2(conv2)
        conv2 = shuffle_utils.gelu(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.norm3(conv3)
        conv3 = shuffle_utils.gelu(conv3)

        in_layer = self.input_linear(conv3) * 0.25

        benes_block = self.benes_block(in_layer, training=training)

        return self.output_linear(benes_block[:, 0, 0, :])
