from enum import Enum

import numpy as np
import tensorflow as tf

import utils.data as data_utils
import utils.shuffle as shuffle_utils
from layers.shuffle import LinearTransform, QuaternarySwitchUnit


class ShuffleType(Enum):
    LEFT = shuffle_utils.qrol
    RIGHT = shuffle_utils.qror

    def __call__(self, *args, **kwargs):
        self.value(*args)


class QuaternaryShuffleLayer(tf.keras.layers.Layer):
    """ Implements quaternary cyclic shift for input tensor.
    """

    def __init__(self, shuffle_type: ShuffleType, layer_level=0, **kwargs):
        super(QuaternaryShuffleLayer, self).__init__(trainable=False, **kwargs)
        self.level = layer_level
        self.shuffled_indices = None
        self.shuffle = shuffle_type

    def build(self, input_shape: tf.TensorShape):
        _, length, _ = input_shape.as_list()
        digits = shuffle_utils.quaternary_digits(length - 1)
        self.shuffled_indices = [self.shuffle(x, digits, self.level) for x in range(length)]

    def call(self, inputs, **kwargs):
        return tf.gather(inputs, self.shuffled_indices, axis=1)


class ZOrderFlatten(tf.keras.layers.Layer):
    """ Implements flattening according to Z-Order curve.
    """

    def call(self, inputs, **kwargs):
        batch_size, width, height, *channels = inputs.shape.as_list()
        vec_size = width * height

        matrix = np.reshape(np.arange(vec_size), [width, height]).tolist()
        quaternary_mask = data_utils.matrix_to_vector(matrix)

        inputs = tf.reshape(inputs, [batch_size, vec_size] + channels)
        return tf.gather(inputs, quaternary_mask, axis=1)


class ZOrderUnflatten(tf.keras.layers.Layer):
    """ Implements vector reshaping to matrix according to Z-Order curve.
    """

    def call(self, inputs, width=None, height=None, **kwargs):
        _, length, nmaps = inputs.shape.as_list()

        matrix = data_utils.vector_to_matrix([x for x in range(length)])
        quaternary_mask = np.reshape(np.array(matrix), [length])

        gather = tf.gather(inputs, quaternary_mask, axis=1)
        return tf.reshape(gather, [-1, width, height, nmaps])


class BenesBlock(tf.keras.layers.Layer):
    """Implementation of Quaternary Beneš block
    This implementation expects 4-D inputs - [batch_size, width, height, channels]
    Output shape will be same as input shape, expect channels will be in size of num_units.
    BenesBlock output is output from the last BenesBlock layer. No additional output processing is applied.
    """

    def __init__(self, block_count, num_units, **kwargs):
        """
        :param block_count: Determines Beneš block count that are chained together
        :param fixed_shuffle: Use fixed shuffle (equal in every layer) or dynamic (shuffle differs in every layer)
        :param num_units: Num units to use in Beneš block
        """
        super().__init__(**kwargs)
        self.block_count = block_count
        self.num_units = num_units

        self.block_layers = None
        self.output_layer = None

    def build(self, input_shape):

        self.block_layers = {}
        for i in range(self.block_count):
            self.block_layers[i] = {
                "forward": QuaternarySwitchUnit("forward", dropout_rate=0.1),
                "middle": QuaternarySwitchUnit("middle", dropout_rate=0.1),
                "reverse": QuaternarySwitchUnit("reverse", dropout_rate=0.1)
            }

        self.output_layer = LinearTransform("output", 1, self.num_units)

    def call(self, inputs, training=False, **kwargs):
        input_shape = inputs.get_shape().as_list()
        level_count = (input_shape[1] - 1).bit_length() - 1

        last_layer = ZOrderFlatten()(inputs)

        for block_nr in range(self.block_count):

            with tf.name_scope(f"benes_block_{block_nr}"):
                for _ in range(level_count):
                    switch = self.block_layers[block_nr]["forward"](last_layer, training=training)
                    last_layer = QuaternaryShuffleLayer(ShuffleType.RIGHT)(switch)

                for level in range(level_count):
                    last_layer = self.block_layers[block_nr]["reverse"](last_layer, training=training)
                    last_layer = QuaternaryShuffleLayer(ShuffleType.LEFT)(last_layer)

                last_layer = self.block_layers[block_nr]["middle"](last_layer, training=training)

        return ZOrderUnflatten()(last_layer, width=input_shape[1], height=input_shape[2])