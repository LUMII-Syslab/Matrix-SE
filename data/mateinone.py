import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data.base import Dataset


class MateInOne(Dataset):
    def __init__(self) -> None:
        super().__init__({
            "input_classes": 13,
            "output_classes": 3,
            "eval_size": 10000,
            "train_lengths": [8],
            "eval_lengths": [8]
        })

        folder = Path(config.data_dir) / "chess"
        self.train_file = folder / "train_set_1move_amb2.nn"
        self.eval_file = folder / "test_set_1move_deterministic2.nn"

        assert self.train_file.exists(), f"Train file {self.train_file} not found"
        assert self.eval_file.exists(), f"Train file {self.eval_file} not found"

    @staticmethod
    def read_dataset(filename):
        with open(filename, 'rb') as fp:
            data_set_loaded = np.array(pickle.load(fp))

        def __generator():
            for x, y in data_set_loaded:
                yield x, y

        return tf.data.Dataset.from_generator(__generator, (tf.int32, tf.int32), ([8, 8], [8, 8]))

    def train_dataset(self) -> list:
        dataset = self.read_dataset(self.train_file)
        return [dataset]

    def eval_dataset(self) -> list:
        dataset = self.read_dataset(self.eval_file)
        return [dataset]
