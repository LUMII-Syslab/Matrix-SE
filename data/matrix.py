from abc import abstractmethod

import numpy as np
import tensorflow as tf

import utils.data as data_utils
from data.base import GeneratorDataset


class MatrixBase(GeneratorDataset):

    def __init__(self) -> None:
        super().__init__({
            "input_classes": 10,
            "output_classes": 10,
            "train_lengths": [4, 8, 16, 32],
            "eval_lengths": [4, 8, 16, 32]
        })

    @abstractmethod
    def label_fn(self, feature: np.ndarray):
        pass

    def generator_fn(self, feature_shape, label_shape, training=False):
        def _generator():
            while True:
                min_l = np.random.choice([1, feature_shape[0] // 2], [2])
                width = np.random.randint(min_l[0], feature_shape[0])
                height = np.random.randint(min_l[1], feature_shape[0])
                feature = self.feature_fn(height, width)
                label = self.label_fn(feature)

                feature = data_utils.pad_with_zeros(feature, feature_shape)
                label = data_utils.pad_with_zeros(label, feature_shape)

                yield feature, label

        return _generator

    def feature_fn(self, height, width):
        return np.random.randint(1, self.config["input_classes"], [width, height])

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
        return 500000

    @property
    def eval_size(self) -> int:
        return 10000


class Identity(MatrixBase):

    def label_fn(self, feature: np.ndarray):
        return feature


class Transpose(MatrixBase):

    def label_fn(self, feature: np.ndarray):
        return np.transpose(feature)


class Rotate90(MatrixBase):

    def label_fn(self, feature: np.ndarray):
        return np.rot90(feature)

    @property
    def train_size(self) -> int:
        return 1000000


class Squaring(MatrixBase):

    def __init__(self) -> None:
        super().__init__()
        self.add_config("input_classes", 3)
        self.add_config("output_classes", 3)
        self.add_config("train_lengths", [4, 8, 16, 32])
        self.add_config("eval_lengths", [4, 8, 16, 32, 64])

    def feature_fn(self, height, width):
        return np.random.randint(1, self.config["input_classes"], [width, width])

    def label_fn(self, feature: np.ndarray):
        ft = feature - 1
        label = np.linalg.matrix_power(ft, 2)
        label = np.mod(label, 2)
        return label.astype(np.int32) + 1

    @property
    def train_size(self) -> int:
        return 1000000
