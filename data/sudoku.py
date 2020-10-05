import urllib.request
import zipfile
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


def randomise_position(x, y):
    randomise_pos = tf.random.uniform(()) > 0.5
    return tf.cond(randomise_pos, lambda: random_padding(x, y), lambda: padding(x, y))


def transpose(x, y):
    return tf.matrix_transpose(x), tf.matrix_transpose(y)


def cycle_row_blocks(sudoku):
    first = sudoku[0:3, :]
    second = sudoku[3:6, :]
    third = sudoku[6:9, :]

    return tf.concat([third, first, second], axis=0)


def cycle_col_blocks(sudoku):
    first = sudoku[:, 0:3]
    second = sudoku[:, 3:6]
    third = sudoku[:, 6:9]

    return tf.concat([third, first, second], axis=1)


def cycle_row_example(x, y):
    return cycle_row_blocks(x), cycle_row_blocks(y)


def cycle_col_example(x, y):
    return cycle_col_blocks(x), cycle_col_blocks(y)


class Sudoku(Dataset):
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
    def __init__(self) -> None:
        super().__init__({
            "input_classes": 11,
            "output_classes": 10,
            "eval_size": 20000,
            "train_lengths": [16],
            "eval_lengths": [16]
        })

        self.folder = Path(config.data_dir) / "sudoku"

    @property
    def train_file(self) -> Path:
        return self.folder / "sudoku_train.csv"

    @property
    def validation_file(self) -> Path:
        return self.folder / "sudoku_val.csv"

    @property
    def eval_file(self) -> Path:
        return self.folder / "sudoku_test.csv"

    @staticmethod
    def read_dataset(csv_file: Path) -> tf.data.Dataset:
        if not csv_file.exists():
            raise FileNotFoundError(f"Train file {csv_file} not found")

        def to_int_list(x, shift: int = 1):
            x = tf.strings.bytes_split(x)
            x = tf.strings.to_number(x, out_type=tf.int32) + shift
            return tf.reshape(x, [9, 9])

        dataset = tf.data.experimental.CsvDataset(str(csv_file), [tf.string, tf.string])
        return dataset.map(lambda x, y: (to_int_list(x), to_int_list(y, shift=0)), num_parallel_calls=AUTOTUNE)

    def train_dataset(self) -> list:
        dataset = self.read_dataset(self.train_file)
        dataset = self.augment_dataset(dataset)
        return [dataset.map(padding, num_parallel_calls=AUTOTUNE)]

    @staticmethod
    def augment_dataset(dataset: tf.data.Dataset):
        """
        Bases on Sudoku isomorphism: https://en.wikipedia.org/wiki/Mathematics_of_Sudoku
        """

        for _ in range(2):
            shuffle_rows = dataset.map(cycle_row_example, num_parallel_calls=AUTOTUNE)
            dataset = dataset.concatenate(shuffle_rows)

        for _ in range(2):
            shuffle_col = dataset.map(cycle_col_example, num_parallel_calls=AUTOTUNE)
            dataset = dataset.concatenate(shuffle_col)

        tran = dataset.map(transpose, num_parallel_calls=AUTOTUNE)
        return dataset.concatenate(tran)

    def eval_dataset(self) -> list:
        dataset = self.read_dataset(self.eval_file)
        dataset = dataset.map(padding, num_parallel_calls=AUTOTUNE)
        return [dataset]


class SudokuHard(Sudoku):
    """
    Dataset from "Recurrent Relational Networks" by Rasmus Berg Palm, Ulrich Paquet,
    Ole Winther
    """
    url = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"
    zip_name = "sudoku-hard.zip"

    def __init__(self) -> None:
        super(SudokuHard, self).__init__()

        folder = Path(config.data_dir)
        if folder.exists():
            print("Downloading data...")
            zip_path = folder / self.zip_name

            if not zip_path.exists():
                urllib.request.urlretrieve(self.url, str(zip_path))
                with zipfile.ZipFile(str(zip_path)) as f:
                    f.extractall(str(folder))

        self.folder_hard = folder / 'sudoku-hard'

    @staticmethod
    def random_mask(x, y):
        mask = tf.random.uniform(y.shape)
        mask = tf.cast(tf.math.greater(mask, 0.5), dtype=tf.int32)
        x = y * mask
        return x, y

    def train_dataset(self) -> list:
        dataset_hard = self.read_dataset(self.train_file)
        dataset_hard = self.augment_dataset(dataset_hard)
        return [dataset_hard.map(padding, num_parallel_calls=AUTOTUNE)]

    def eval_dataset(self) -> list:
        dataset_hard = self.read_dataset(self.eval_file)
        dataset_hard = dataset_hard.map(padding, num_parallel_calls=AUTOTUNE)
        return [dataset_hard]

    @property
    def train_file(self) -> Path:
        return self.folder_hard / 'train.csv'

    @property
    def validation_file(self) -> Path:
        return self.folder_hard / 'valid.csv'

    @property
    def eval_file(self) -> Path:
        return self.folder_hard / 'test.csv'
