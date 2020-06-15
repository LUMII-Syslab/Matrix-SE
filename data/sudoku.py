"""
@misc{sudoku2018,
  author = {Park, Kyubyong},
  title = {Can Convolutional Neural Networks Crack Sudoku Puzzles?},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/Kyubyong/sudoku}}
}
"""
from pathlib import Path

import tensorflow as tf

import config
from data.base import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE


def random_padding(x, y):
    hpad_b = tf.random.uniform((), minval=0, maxval=8, dtype=tf.int32)
    hpad_a = 7 - hpad_b

    wpad_b = tf.random.uniform((), minval=0, maxval=8, dtype=tf.int32)
    wpad_a = 7 - wpad_b
    x = tf.pad(x, [[hpad_b, hpad_a], [wpad_b, wpad_a]])
    y = tf.pad(y, [[0, 7], [0, 7]])
    x = tf.ensure_shape(x, [16, 16])
    y = tf.ensure_shape(y, [16, 16])

    return x, y


def padding(x, y):
    x = tf.pad(x, [[0, 7], [0, 7]])
    y = tf.pad(y, [[0, 7], [0, 7]])

    return x, y


class Sudoku(Dataset):
    def __init__(self) -> None:
        super().__init__({
            "input_classes": 12,
            "output_classes": 12,
            "eval_size": 10000,
            "train_lengths": [16],
            "eval_lengths": [16]
        })

        folder = Path(config.data_dir) / "sudoku"
        self.train_file = folder / "sudoku_train.csv"
        self.eval_file = folder / "sudoku_test.csv"

        assert self.train_file.exists(), f"Train file {self.train_file} not found"
        assert self.eval_file.exists(), f"Test file {self.eval_file} not found"

    @staticmethod
    def read_dataset(csv_file):
        def to_int_list(x):
            x = tf.strings.bytes_split(x)
            x = tf.strings.to_number(x, out_type=tf.int32) + 1
            return tf.reshape(x, [9, 9])

        dataset = tf.data.experimental.CsvDataset(str(csv_file), [tf.string, tf.string])
        return dataset.map(lambda x, y: (to_int_list(x), to_int_list(y)), num_parallel_calls=AUTOTUNE)

    @staticmethod
    def augment_dataset(dataset: tf.data.Dataset):
        # place sudoku puzzle randomly on larger padding matrix
        return dataset.map(random_padding, num_parallel_calls=AUTOTUNE)

    def train_dataset(self) -> list:
        dataset = self.read_dataset(self.train_file)
        return [self.augment_dataset(dataset)]

    def eval_dataset(self) -> list:
        dataset = self.read_dataset(self.eval_file)
        dataset = dataset.map(padding, num_parallel_calls=AUTOTUNE)
        return [dataset]
