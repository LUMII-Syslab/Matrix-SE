import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Dense, Embedding
from tensorflow.keras.layers import LayerNormalization

from models.base import Model


class ResNet(Model):
    """
    ResNet implementation of "Identity Mappings in Deep Residual Networks" by
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

    We replace batch normalization with layer normalization as we found that such
    model works better for algorithmic tasks.
    Link: https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(self, feature_maps: int, kernel_size: int, residual_blocks: int) -> None:
        super().__init__()
        self.feature_maps = feature_maps
        self.kernel_size = kernel_size
        self.tower_size = residual_blocks

    def build(self, input_classes, output_classes):
        self.embedding = Embedding(input_classes, self.feature_maps, name="embedding")

        self.start_conv = Conv2D(self.feature_maps, self.kernel_size, padding="same")

        self.residual_tower = [
            ResidualBlock(self.feature_maps, self.kernel_size, name=f"res_block_{i}") for i in
            range(self.tower_size)]

        self.norm = LayerNormalization(axis=-1, name="last_norm")
        self.linear = Dense(output_classes, name="linear")

    def call(self, inputs, training=False):
        emb = self.embedding(inputs)
        emb = tf.identity(emb, name="input_node")

        last = self.start_conv(emb)

        for block in self.residual_tower:
            last = block(last, training=training)

        last = tf.identity(last, name="output_node")
        last = self.norm(last, training=training)
        last = tf.nn.relu(last)

        return self.linear(last)


class ResidualBlock(Layer):

    def __init__(self, feature_maps, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.feature_maps = feature_maps
        self.kernel_size = kernel_size
        self.norm1 = LayerNormalization(axis=-1, name="norm_1")
        self.conv1 = Conv2D(self.feature_maps, self.kernel_size, padding="same",
                            name=f"conv_{self.kernel_size}x{self.kernel_size}_first")

        self.norm2 = LayerNormalization(axis=-1, name="norm_2")
        self.conv2 = Conv2D(self.feature_maps, self.kernel_size, padding="same",
                            name=f"conv_{self.kernel_size}x{self.kernel_size}_second")

    def call(self, inputs, training=False, **kwargs):
        norm = self.norm1(inputs, training=training)
        norm = tf.nn.relu(norm)
        conv = self.conv1(norm)

        norm = self.norm2(conv, training=training)
        norm = tf.nn.relu(norm)
        conv = self.conv2(norm)

        return tf.add(inputs, conv)
