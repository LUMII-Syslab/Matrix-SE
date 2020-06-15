import numpy as np
import tensorflow as tf

from data.base import Dataset


class CIFAR10(Dataset):
    def __init__(self) -> None:
        super().__init__({
            "input_classes": None,
            "output_classes": 10,
            "train_lengths": [32],
            "eval_lengths": [32],
        })
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train / 128 - 1
        x_test = x_test / 128 - 1

        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)

        self.train = (x_train.astype(np.float32), y_train.astype(np.int32))
        self.eval = (x_test.astype(np.float32), y_test.astype(np.int32))

    def train_dataset(self) -> list:
        data = tf.data.Dataset.from_tensor_slices(self.train)
        data = data.map(lambda x, y: (tf.image.random_flip_left_right(x), y))  # Augment training dataset
        return [data]

    def eval_dataset(self) -> list:
        data = tf.data.Dataset.from_tensor_slices(self.eval)
        return [data]
