import tensorflow as tf

import config
from data.base import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetBuilder:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def make_train_fn(self, custom_params: dict = None):
        def input_fn(params: dict):
            if custom_params:
                params.update(custom_params)

            datasets = self.dataset.train_dataset()
            return self.__prepare_dataset(datasets, params)

        return input_fn

    def make_eval_fn(self, custom_params: dict = None):
        def input_fn(params: dict):
            if custom_params:
                params.update(custom_params)

            datasets = self.dataset.eval_dataset()
            return self.__prepare_dataset(datasets, params)

        return input_fn

    def dataset_name(self):
        return str(self.dataset.__class__.__name__).lower()

    def make_test_fn(self, custom_params: dict = None):
        def input_fn(params: dict):
            if custom_params:
                params.update(custom_params)

            datasets = self.dataset.test_dataset()
            return self.__prepare_test_dataset(datasets, params)

        return input_fn

    @staticmethod
    def split_dataset(*x):
        inputs_stack = []
        target_stack = []
        for inputs, targets in x:
            inputs_stack.append(inputs)
            target_stack.append(targets)

        return tuple(inputs_stack), tuple(target_stack)

    def __prepare_dataset(self, datasets: list, params: dict):
        """
        :param datasets: List of tensorflow.data.Dataset elements
        :return: tensorflow.data.Dataset
        """
        datasets = [data.shuffle(config.shuffle_size) for data in datasets]
        datasets = [data.batch(params['batch_size'], drop_remainder=True) for data in datasets]
        datasets = [data.repeat() for data in datasets]

        output_dataset = tf.data.Dataset.zip(tuple(datasets))
        output_dataset = output_dataset.map(self.split_dataset)

        return output_dataset.prefetch(AUTOTUNE)

    def __prepare_test_dataset(self, datasets, params):
        """
        :param datasets: List of tensorflow.data.Dataset elements
        :return: tensorflow.data.Dataset
        """
        datasets = [data.batch(params['batch_size'], drop_remainder=True) for data in datasets]

        output_dataset = tf.data.Dataset.zip(tuple(datasets))
        output_dataset = output_dataset.map(self.split_dataset)

        return output_dataset.prefetch(AUTOTUNE)
