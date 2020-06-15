from enum import Enum

import numpy as np
import tensorflow as tf

import config
import utils.data as data_utils
import utils.shuffle as shuffle_utils
from utils.log_memory import log_in_tb


class LinearTransform(tf.keras.layers.Layer):
    """
    Linear Transformation layer along feature dimension.
    """

    def __init__(self, name, n_out, bias_start=0.0, init_scale=1.0, add_bias=True, **kwargs):
        super(LinearTransform, self).__init__(trainable=True, name=name, **kwargs)
        self.n_out = n_out
        self.bias_start = bias_start
        self.init_scale = init_scale
        self.kernel = None
        self.bias_term = None
        self.n_in = None
        self.add_bias = add_bias

    def build(self, input_shape: tf.TensorShape):
        self.n_in = input_shape.as_list()[-1]

        initializer = tf.variance_scaling_initializer(scale=self.init_scale, mode="fan_avg", distribution="uniform")
        self.kernel = self.add_weight("CvK", [self.n_in, self.n_out], initializer=initializer)

        if self.add_bias:
            self.bias_term = self.add_weight("CvB", [self.n_out], initializer=tf.constant_initializer(self.bias_start))

    def call(self, inputs, **kwargs):

        input_shape = inputs.get_shape().as_list()

        in_shape = 1
        for shape in input_shape[:-1]:
            in_shape *= shape

        reshape_in = [in_shape, self.n_in]
        reshape_out = [shape for shape in input_shape[:-1]] + [self.n_out]

        res = tf.matmul(tf.reshape(inputs, reshape_in), self.kernel)
        res = tf.reshape(res, reshape_out)

        if self.add_bias:
            res = res + self.bias_term

        return res


class QuaternarySwitchUnit(tf.keras.layers.Layer):
    """
    Quaternary Switch Unit with 4 inputs and 4 outputs.
    Bases on Residual Switch Unit (https://arxiv.org/abs/2004.04662).
    """

    def __init__(self, name, channel_count=4, dropout_rate=0.1, **kwargs):
        super(QuaternarySwitchUnit, self).__init__(name=name, **kwargs)
        self.channel_count = channel_count
        self.dropout_rate = dropout_rate
        self.residual_weight = 0.9
        self.candidate_weight = np.sqrt(1 - self.residual_weight ** 2) * 0.25
        self.scale_init = np.log(self.residual_weight / (1 - self.residual_weight))

        self.num_units = None
        self.reshaped_units = None
        self.residual_scale = None
        self.layer_norm = None
        self.dropout = None

        self.linear_one = None
        self.linear_two = None

    def build(self, input_shape):
        self.num_units = input_shape.as_list()[2]
        self.reshaped_units = self.num_units * self.channel_count

        initializer = tf.constant_initializer(self.scale_init)
        self.residual_scale = self.add_weight("residual", [self.reshaped_units], initializer=initializer)

        self.linear_one = LinearTransform("linear_one", self.reshaped_units * 2, add_bias=False)
        self.linear_two = LinearTransform("linear_two", self.reshaped_units)

        # Log in TensorBoard
        if config.log_layer_outputs:
            self.linear_one = log_in_tb(self.linear_one, f"input_gate")
            self.linear_two = log_in_tb(self.linear_two, f"output_gate")

        self.layer_norm = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, **kwargs):
        batch_size, length, num_units = inputs.shape.as_list()[:3]
        # reshape input - each new element contains 4 previous elements along feature dimension
        inputs = tf.reshape(inputs, [batch_size, length // self.channel_count, self.reshaped_units])
        dropout = self.dropout(inputs, training=training)

        first_linear = self.linear_one(dropout)
        norm = self.layer_norm(first_linear)
        gelu = shuffle_utils.gelu(norm)
        second_linear = self.linear_two(gelu)

        residual_scale = tf.nn.sigmoid(self.residual_scale)

        candidate = residual_scale * inputs + second_linear * self.candidate_weight
        return tf.reshape(candidate, [batch_size, length, self.num_units])

    @staticmethod
    def shuffle_two_axis(inputs):
        with tf.variable_scope('shuffle_inputs'):
            length = inputs.shape[1]
            num_units = inputs.shape[-1]
            row_sfl = [x ^ 1 for x in range(length)]
            col_sfl = [x ^ 2 for x in range(length)]
            row_col_sfl = [(x ^ 1) ^ 2 for x in range(length)]

            identity = inputs[:, :, :num_units // 4]
            shuffled_rows = tf.gather(inputs[:, :, num_units // 4: num_units // 2], row_sfl, axis=1)
            shuffled_col = tf.gather(inputs[:, :, num_units // 2:num_units // 2 + num_units // 4], col_sfl, axis=1)
            shuffled_row_col = tf.gather(inputs[:, :, num_units // 2 + num_units // 4:], row_col_sfl, axis=1)

            return tf.concat([identity, shuffled_rows, shuffled_col, shuffled_row_col], axis=2)


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer Normalization without output bias and gain. https://arxiv.org/abs/1911.07013
    """

    def __init__(self, axis=1, epsilon=1e-3, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        num_units = input_shape.as_list()[-1]
        self.bias = self.add_weight("bias", [1, 1, num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, **kwargs):
        inputs -= tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        inputs += self.bias
        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)


class ShuffleType(Enum):
    """
    Implementation of left and right permutation.
    """
    LEFT = shuffle_utils.qrol
    RIGHT = shuffle_utils.qror

    def __call__(self, *args, **kwargs):
        self.value(*args)


class QuaternaryShuffle(tf.keras.layers.Layer):
    """
        Permutes input elements according to cyclic rotation of base-4 numbers.
    """

    def __init__(self, shuffle_type: ShuffleType, layer_level=0, **kwargs):
        super(QuaternaryShuffle, self).__init__(trainable=False, **kwargs)
        self.level = layer_level
        self.shuffled_indices = None
        self.shuffle = shuffle_type

    def build(self, input_shape: tf.TensorShape):
        _, length, _ = input_shape.as_list()
        digits = shuffle_utils.quaternary_digits(length - 1)
        self.shuffled_indices = [self.shuffle(x, digits, self.level) for x in range(length)]

    def call(self, inputs, **kwargs):
        return tf.gather(inputs, self.shuffled_indices, axis=1)


class QuaternaryFlatten(tf.keras.layers.Layer):
    """
        Flatten 2D array to sequence according to Z-order-curve.
    """

    def call(self, inputs, **kwargs):
        batch_size, width, height, *channels = inputs.shape.as_list()
        vec_size = width * height

        matrix = np.reshape(np.arange(vec_size), [width, height]).tolist()
        quaternary_mask = data_utils.matrix_to_vector(matrix)

        inputs = tf.reshape(inputs, [batch_size, vec_size] + channels)
        return tf.gather(inputs, quaternary_mask, axis=1)


class QuaternaryReshape(tf.keras.layers.Layer):
    """
    Reshapes sequence of elements to 2D representation according to Z-order-curve.
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

        self.output_layer = LinearTransform("output", self.num_units)

    def call(self, inputs, training=False, **kwargs):
        input_shape = inputs.get_shape().as_list()
        level_count = (input_shape[1] - 1).bit_length() - 1

        last_layer = QuaternaryFlatten()(inputs)

        for block_nr in range(self.block_count):

            with tf.name_scope(f"benes_block_{block_nr}"):
                for _ in range(level_count):
                    switch = self.block_layers[block_nr]["forward"](last_layer, training=training)
                    last_layer = QuaternaryShuffle(ShuffleType.RIGHT)(switch)

                self.show_as_2d("exchange_one", input_shape, last_layer)

                for level in range(level_count):
                    last_layer = self.block_layers[block_nr]["reverse"](last_layer, training=training)
                    last_layer = QuaternaryShuffle(ShuffleType.LEFT)(last_layer)

                self.show_as_2d("exchange_two", input_shape, last_layer)

                last_layer = self.block_layers[block_nr]["middle"](last_layer, training=training)
                self.show_as_2d("benes_middle", input_shape, last_layer)

        return QuaternaryReshape()(last_layer, width=input_shape[1], height=input_shape[2])

    def show_as_2d(self, name, input_shape, last_layer):
        if config.log_2d_outputs:
            image = QuaternaryReshape()(last_layer, width=input_shape[1], height=input_shape[2])
            image = tf.transpose(image, [3, 1, 2, 0])
            tf.compat.v1.summary.image(name, image[:, :, :, :1], max_outputs=4, family="shuffle-exchange")
