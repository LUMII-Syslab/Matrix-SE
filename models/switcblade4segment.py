import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D

from layers.shuffle import LinearTransform, LayerNormalization, BenesBlock
from models.base import Model
from utils.shuffle import gelu


class Switchblade4Segment(Model):
    """
    Switchblade with convolutional layers used for CityScapes
    pixel-level semantic segmentation.
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

        self.deconv = None
        self.deconv2 = None
        self.deconv3 = None

        self.output_linear = None

    def build(self, input_classes, output_classes):
        # Squeeze image to square, so Switchblade can process
        self.conv = Conv2D(self.num_units // 2, 4, strides=(1, 2), padding="same", use_bias=False)
        self.norm1 = LayerNormalization(axis=[1, 2])

        self.conv2 = Conv2D(self.num_units, 4, strides=(2, 2), padding="same", use_bias=False)
        self.norm2 = LayerNormalization(axis=[1, 2])

        self.conv3 = Conv2D(self.num_units * 2, 4, strides=(2, 2), padding="same", use_bias=False)
        self.norm3 = LayerNormalization(axis=[1, 2])

        self.input_linear = LinearTransform("input", self.num_units)
        self.benes_block = BenesBlock(self.block_count, self.num_units)

        self.deconv = Conv2DTranspose(self.num_units * 2, 4, strides=(2, 2), padding="same", use_bias=False)
        self.deconv2 = Conv2DTranspose(self.num_units, 4, strides=(2, 2), padding="same", use_bias=False)
        self.deconv3 = Conv2DTranspose(self.num_units // 2, 4, strides=(1, 2), padding="same", use_bias=False)

        self.output_linear = LinearTransform("output", output_classes)

    def call(self, inputs, training=False):
        conv1 = self.conv(inputs)
        conv1 = self.norm1(conv1)
        conv1 = gelu(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.norm2(conv2)
        conv2 = gelu(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.norm3(conv3)
        conv3 = gelu(conv3)

        in_layer = self.input_linear(conv3) * 0.25
        benes_block = self.benes_block(in_layer, training=training)

        # Add U-Net style skip connections
        in_concat = tf.concat([benes_block, in_layer], axis=3)
        out_layer = self.deconv(in_concat)
        out_layer = gelu(out_layer)

        in_concat = tf.concat([out_layer, conv2], axis=3)
        out_layer = self.deconv2(in_concat)
        out_layer = gelu(out_layer)

        in_concat = tf.concat([out_layer, conv1], axis=3)
        out_layer = self.deconv3(in_concat)
        out_layer = gelu(out_layer)

        return self.output_linear(out_layer)
