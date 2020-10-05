import numpy as np
import tensorflow as tf

import config
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
    Based on Residual Switch Unit (https://arxiv.org/abs/2004.04662).
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

        if config.log_layer_outputs:
            self.linear_one = log_in_tb(self.linear_one, f"input_gate")
            self.linear_two = log_in_tb(self.linear_two, f"output_gate")

        self.layer_norm = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, **kwargs):
        batch_size, length, num_units = inputs.shape.as_list()[:3]
        inputs = tf.reshape(inputs, [batch_size, length // self.channel_count, self.reshaped_units])
        dropout = self.dropout(inputs, training=training)

        first_linear = self.linear_one(dropout)
        norm = self.layer_norm(first_linear)
        gelu = shuffle_utils.gelu(norm)
        second_linear = self.linear_two(gelu)

        residual_scale = tf.nn.sigmoid(self.residual_scale)

        candidate = residual_scale * inputs + second_linear * self.candidate_weight
        return tf.reshape(candidate, [batch_size, length, self.num_units])


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, axis=1, epsilon=1e-6, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        super(LayerNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)
