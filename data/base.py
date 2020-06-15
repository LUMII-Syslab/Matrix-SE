import itertools
from abc import abstractmethod, ABCMeta
from pathlib import Path

import tensorflow as tf

import config


class Dataset(metaclass=ABCMeta):

    def __init__(self, config: dict = None) -> None:
        self.__config = config if config else {}

    @abstractmethod
    def train_dataset(self) -> list:
        """
        :return: List of tensorflow.data.Dataset elements
        """
        pass

    @abstractmethod
    def eval_dataset(self) -> list:
        """
        :return: List of tensorflow.data.Dataset elements
        """
        pass

    def add_config(self, key: str, value):
        self.__config[key] = value

    def add_configs(self, config: dict):
        self.__config.update(config)

    @property
    def config(self) -> dict:
        return self.__config

    def test_dataset(self):
        """ Used for testing on whole dataset
        :return: List of tensorflow.data.Dataset elements
        """
        return self.eval_dataset()


class GeneratorDataset(Dataset):
    @abstractmethod
    def generator_fn(self, feature_shape, label_shape, training=False) -> callable:
        """ Returns generator function that generates tuple of feature and label.

        Generator should return result matching output types and output_shape.

        :param training: Is data generated for training
        :param label_shape: generator output shape for feature
        :param feature_shape: generator output shape for label
        :return: generator function
        """
        pass

    @property
    @abstractmethod
    def generator_output_types(self):
        """
        Shape of output using tf data types.
        :return: Tuple
        """
        pass

    @property
    @abstractmethod
    def train_output_shapes(self) -> list:
        """
        Each tuple in list represent generator output shape (for feature and label)
        :return: List of tuples
        """
        pass

    @property
    @abstractmethod
    def eval_output_shapes(self) -> list:
        """
        Each tuple in list represent generator output shape.
        :return: List of tuples
        """
        pass

    @property
    @abstractmethod
    def train_size(self) -> int:
        pass

    @property
    @abstractmethod
    def eval_size(self) -> int:
        pass

    def train_dataset(self) -> list:
        return self.create_file_based_dataset("train", self.train_output_shapes, self.train_size, training=True)

    def create_file_based_dataset(self, file_prefix: str, output_shapes: list, dataset_size, training):
        data_dir = Path(config.data_dir) / self.__class__.__name__.lower()

        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        datasets = []
        for feature_sh, label_sh in output_shapes:
            f_sh_str = "x".join([str(x) for x in feature_sh])
            file_name = f"{file_prefix}_{f_sh_str}.tfrecord"
            file_name = data_dir / file_name

            data = self.dataset_from_file(file_name, feature_sh, label_sh, dataset_size, training)

            datasets.append(data)
        return datasets

    def dataset_from_file(self, file_name: Path, feature_sh: tuple,
                          label_sh: tuple, dataset_size, training: bool) -> tf.data.TFRecordDataset:

        if file_name.exists() and config.force_file_generation:
            file_name.unlink()

        if not file_name.exists():
            generator = self.generator_fn(feature_sh, label_sh, training=training)
            generator = itertools.islice(generator(), dataset_size)
            options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)

            with tf.io.TFRecordWriter(str(file_name), options) as tfwriter:
                for feature, label in generator:
                    example = self.create_example(feature, label)
                    tfwriter.write(example.SerializeToString())

        data = tf.data.TFRecordDataset([str(file_name)], "GZIP")
        return data.map(lambda rec: self.extract(rec, feature_sh, label_sh), tf.data.experimental.AUTOTUNE)

    @staticmethod
    def extract(data_record, feature_shape, label_shape):
        parsed = tf.io.parse_single_example(data_record, {
            'feature': tf.io.VarLenFeature(tf.int64),
            'label': tf.io.VarLenFeature(tf.int64),
        })

        feature = tf.reshape(parsed["feature"].values, feature_shape)  # type: tf.Tensor
        label = tf.reshape(parsed["label"].values, label_shape)  # type: tf.Tensor

        feature.set_shape(feature_shape)
        label.set_shape(label_shape)

        feature = tf.cast(feature, tf.int32)
        label = tf.cast(label, tf.int32)

        return feature, label

    @staticmethod
    def create_example(feature, label):
        feature = tf.train.Int64List(value=feature.flatten())
        label = tf.train.Int64List(value=label.flatten())

        example_map = {
            'feature': tf.train.Feature(int64_list=feature),
            'label': tf.train.Feature(int64_list=label),
        }

        features = tf.train.Features(feature=example_map)
        return tf.train.Example(features=features)

    def eval_dataset(self) -> list:
        return self.create_file_based_dataset("eval", self.eval_output_shapes, self.eval_size, training=False)

    def create_dataset(self, feature_sh, label_sh):
        return tf.data.Dataset.from_generator(
            self.generator_fn(feature_sh, label_sh, training=False),
            self.generator_output_types,
            output_shapes=(tf.TensorShape(feature_sh), tf.TensorShape(label_sh))
        )
