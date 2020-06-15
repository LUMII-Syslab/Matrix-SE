from abc import abstractmethod

import numpy as np
import tensorflow as tf

import utils.data as data_utils
from data.base import GeneratorDataset


class MatricesBase(GeneratorDataset):

    def __init__(self) -> None:
        super().__init__()

        self.add_configs({
            "input_classes": 3,
            "output_classes": 3,
            "train_lengths": [8, 16, 32],
            "eval_lengths": [8, 16, 32, 64]
        })

    @abstractmethod
    def label_fn(self, first_matrix: np.ndarray, second_matrix: np.ndarray):
        pass

    @abstractmethod
    def second_matrix_fn(self, height, width):
        pass

    @abstractmethod
    def first_matrix_fn(self, height, width):
        pass

    def generator_fn(self, feature_shape, label_shape, training=False) -> callable:
        half_length = (feature_shape[0] - 1) // 2
        length = feature_shape[0]

        def _generator():
            while True:
                height = max(np.random.binomial(half_length, 0.7, 1)[0], 1)
                width = max(np.random.binomial(length, 0.7, 1)[0], 1)
                first_matrix = self.first_matrix_fn(height, width)
                second_matrix = self.second_matrix_fn(height, width)

                label = self.label_fn(first_matrix, second_matrix)

                first_matrix += 2  # Shift by 2 to make space for padding and delimiter
                second_matrix += 2
                label += 1  # Shift by 1 to make space for padding

                # add delimiter between matrices
                delimiter = np.ones([width, 1], dtype=np.int32)
                feature = np.concatenate([first_matrix, delimiter, second_matrix], axis=1)

                feature = data_utils.pad_with_zeros(feature, feature_shape)
                label = data_utils.pad_with_zeros(label, label_shape)

                yield feature, label

        return _generator

    @property
    def generator_output_types(self):
        return tf.int32, tf.int32

    @property
    def train_output_shapes(self) -> list:
        return [((x, x), (x, x)) for x in self.config["train_lengths"]]

    @property
    def eval_output_shapes(self) -> list:
        return [((x, x), (x, x)) for x in self.config["eval_lengths"]]

    @property
    def train_size(self) -> int:
        return 1000000

    @property
    def eval_size(self) -> int:
        return 20000


class BitwiseXOR(MatricesBase):

    def __init__(self) -> None:
        super().__init__()
        number_bits = 1
        self.max_number = 2 ** number_bits
        self.add_config("input_classes", self.max_number + 2)
        self.add_config("output_classes", self.max_number + 1)

    def first_matrix_fn(self, height, width):
        return np.random.randint(0, self.max_number, [width, height], dtype=np.int32)

    def second_matrix_fn(self, height, width):
        return np.random.randint(0, self.max_number, [width, height], dtype=np.int32)

    def label_fn(self, first_matrix: np.ndarray, second_matrix: np.ndarray):
        return np.bitwise_xor(first_matrix, second_matrix)
