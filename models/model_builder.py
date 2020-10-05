from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data.base import Dataset
from models.base import Model
from optimization.radam import RAdamOptimizer
from utils.log_memory import show_layer_outputs


def get_accuracy(prediction, label):
    correct_symbols = tf.equal(prediction, label)
    mask_y_in = tf.cast(tf.not_equal(label, 0), tf.float32)
    mask_out = tf.cast(tf.not_equal(prediction, 0), tf.float32)
    mask_2 = tf.maximum(mask_y_in, mask_out)
    correct_symbols = tf.cast(correct_symbols, tf.float32)
    correct_symbols *= mask_2
    return tf.reduce_sum(correct_symbols) / tf.reduce_sum(mask_2)


def get_all_correct(prediction, label):
    correct_symbols = tf.equal(prediction, label)
    mask_y_in = tf.cast(tf.not_equal(label, 0), tf.float32)
    mask_out = tf.cast(tf.not_equal(prediction, 0), tf.float32)
    mask_2 = tf.maximum(mask_y_in, mask_out)
    correct_symbols = tf.cast(correct_symbols, tf.float32)
    correct_symbols = correct_symbols * mask_2 + (1 - mask_2)
    reduce_indices = tf.range(1, tf.rank(correct_symbols))  # keep only batch dimension
    correct_symbols = tf.reduce_min(correct_symbols, axis=reduce_indices)
    return tf.reduce_mean(correct_symbols)


class AverageLoggingHook(tf.estimator.SessionRunHook):

    def __init__(self, tensors: dict, file_name: Path = None):
        """
        :param tensors: dictionary with tensors for logging
        :param file_name: File to log values
        """
        self._tensors = tensors
        self._total_value = None
        self._steps = None
        self._file_name = file_name

    def begin(self):
        self._total_value = {key: 0 for key in self._tensors.keys()}
        self._steps = 0

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):
        for key, value in run_values.results.items():
            self._total_value[key] += value
        self._steps += 1

    def end(self, session):
        if self._file_name:
            with self._file_name.open("a+") as file:
                self.__log(file.write)

        self.__log(print)

    def __log(self, output_stream):
        for key, value in self._total_value.items():
            output_stream(f"Average {key} over {self._steps} steps: {value / self._steps}\n")


class ModelAssembler:

    def __init__(self, model: Model, dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset

    def assemble_model(self, features, labels, training: bool):
        input_classes = self.dataset.config["input_classes"]
        output_classes = self.dataset.config["output_classes"]

        self.model.build(input_classes, output_classes)

        total_loss = 0.0
        length_predictions = []

        log_dict = {}

        for inputs, targets in zip(features, labels):
            length = inputs.shape.as_list()[1]
            logits_all = self.model.call(inputs, training=training)
            logits_all = logits_all if isinstance(logits_all, list) else [logits_all]

            for logits in logits_all:
                predictions = {
                    'classes': tf.argmax(input=logits, axis=-1, output_type=tf.int32),
                    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
                }

                loss = self.calculate_loss_with_smoothing(targets, logits, output_classes,config.adjust_class_imbalance)
                total_loss += loss
                tf.compat.v1.summary.scalar(f"loss_{length}", loss, family="loss")

            log_dict[f"accuracy_{length}x{length}"] = get_accuracy(predictions["classes"], targets)
            tf.compat.v1.summary.scalar(f"{length}x{length}", log_dict[f"accuracy_{length}x{length}"],
                                        family="accuracy")

            log_dict[f"total_accuracy_{length}x{length}"] = get_all_correct(predictions["classes"], targets)
            tf.compat.v1.summary.scalar(f"total{length}x{length}", log_dict[f"total_accuracy_{length}x{length}"],
                                        family="accuracy")

            length_predictions.append((targets, predictions["classes"]))

        tf.compat.v1.summary.scalar("total_loss", total_loss, family="loss")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            name = var.name.replace("var_lengths", "")
            tf.compat.v1.summary.histogram(name + '/histogram', var)

        gen_file = Path(config.model_dir) / config.gen_test_file
        return total_loss, length_predictions, AverageLoggingHook(log_dict, file_name=gen_file), predictions["classes"]

    @staticmethod
    def calculate_loss(labels, logits, output_classes):
        labels_one_hot = tf.one_hot(labels, output_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
        return tf.reduce_mean(loss)

    def calculate_loss_with_smoothing(self, label, logits, output_classes, adjust_for_class_imbalance=False):
        confidence = 1 - config.label_smoothing
        low_confidence = config.label_smoothing / (output_classes - 1)

        labels_one_hot = tf.one_hot(label, output_classes, on_value=confidence, off_value=low_confidence)

        if adjust_for_class_imbalance:
            # count the number of occurrences of each class in the batch
            class_sum = tf.maximum(tf.reduce_sum(labels_one_hot, axis=tf.range(tf.rank(label))), 1.0)
            class_weights = 1.0 / class_sum
            class_weights = class_weights / tf.reduce_mean(class_weights) * output_classes

            if config.reduce_void_weight:
                # reduce the weight of the padding/void symbol
                select_first = tf.one_hot(0, output_classes, on_value=0., off_value=1., dtype=tf.float32)
                class_weights /= tf.reduce_mean(class_weights)
                class_weights = (class_weights * select_first) * 0.9 + 0.1

            weights = tf.gather(class_weights, label)
        else:
            # reduce the weight of the padding symbol
            mask_out = tf.cast(tf.not_equal(label, 0), tf.float32)
            weights = mask_out * 0.9 + 0.1
            weights /= tf.reduce_mean(weights, keepdims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)

        # the minimum cross_entropy achievable with label smoothing
        min_loss = -(confidence * np.log(confidence) + (output_classes - 1) *
                     low_confidence * np.log(low_confidence + 1e-20))

        ce_loss = tf.reduce_mean((loss - min_loss) * weights)
        return ce_loss

    def log_model_info(self):
        info = "\n\n"
        header = f"-------------------- {self.model.__class__.__name__} --------------------"
        info += header + "\n"
        params = np.sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        info += f"\tTrainable parameters: {params}\n"
        params = np.sum([np.prod(v.shape) for v in tf.compat.v1.global_variables()])
        info += f"\tTotal variables: {params}\n"

        if config.calculate_fcn_receptive_field:
            import receptive_field as rf
            graph = tf.get_default_graph().as_graph_def()
            values = rf.compute_receptive_field_from_graph_def(graph, 'input_node', 'output_node')

            rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y = values
            info += f"\tReceptive field for FCN: {rf_x}x{rf_y}\n"
            info += f"\tEffective stride for FCN: {eff_stride_x}x{eff_stride_y}\n"
            info += f"\tEffective padding for FCN: {eff_pad_x}x{eff_pad_y}\n"

        info += ''.join(["-"] * len(header)) + "\n"

        tf.compat.v1.logging.info(info)


class ModelBuilder(metaclass=ABCMeta):

    @abstractmethod
    def create_model_fn(self):
        pass


def eval_metrics(accuracy_inputs):
    metrics = {}

    for labels, predictions in accuracy_inputs:
        length = labels.shape.as_list()[-2]
        name = f"accuracy_{length}"
        metrics[name] = masked_accuracy(labels, predictions)
        metrics[f"all_correct_{length}"] = masked_all_correct(labels, predictions)

    return metrics


def all_correct(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if labels.dtype != predictions.dtype:
        predictions = tf.cast(predictions, labels.dtype)

    is_correct = tf.cast(tf.equal(predictions, labels), tf.float32)
    reduce_indices = tf.range(1, tf.rank(is_correct))  # keep only batch dimension
    is_correct = tf.reduce_min(is_correct, axis=reduce_indices)
    name = name or 'all_correct'

    return tf.metrics.mean(is_correct, weights, metrics_collections, updates_collections, name)


def masked_accuracy(labels, predictions):
    mask_y_in = tf.not_equal(labels, 0)
    mask_y_in = tf.to_int32(mask_y_in)
    mask_out = tf.not_equal(predictions, 0)
    mask_out = tf.to_int32(mask_out)
    mask_pad = tf.maximum(mask_y_in, mask_out)

    return tf.metrics.accuracy(labels, predictions, weights=mask_pad)


def masked_all_correct(labels, predictions):
    mask_y_in = tf.not_equal(labels, 0)
    mask_y_in = tf.to_int32(mask_y_in)
    mask_out = tf.not_equal(predictions, 0)
    mask_out = tf.to_int32(mask_out)
    mask_pad = tf.maximum(mask_y_in, mask_out)

    return all_correct(labels, predictions, weights=mask_pad)


def summary_hook(training_mode):
    output_dir = config.model_dir + "/" + training_mode
    return tf.estimator.SummarySaverHook(
        save_steps=config.save_checkpoint_steps,
        output_dir=output_dir,
        summary_op=tf.compat.v1.summary.merge_all()
    )


def log_config_in_tb(model_config: dict, dataset_config: dict):
    general_config = [
        ["Learning rate", str(config.learning_rate)],
        ["Train batch size", str(config.train_batch_size)],
        ["Eval batch size", str(config.eval_batch_size)]
    ]
    tf.compat.v1.summary.text("general_config", tf.constant(general_config))

    model_config = [[key, str(val)] for key, val in model_config.items()]
    tf.compat.v1.summary.text("model_config", tf.constant(model_config))

    dataset_config = [[key, str(val)] for key, val in dataset_config.items()]
    tf.compat.v1.summary.text("dataset_config", tf.constant(dataset_config))


class GPUModelBuilder(ModelBuilder):

    def __init__(self, model: Model, dataset: Dataset) -> None:
        self.assembler = ModelAssembler(model, dataset)

    def create_model_fn(self):

        def model_fn(features: tf.data.Dataset, labels: tf.data.Dataset, mode, params):

            is_training = mode == tf.estimator.ModeKeys.TRAIN

            if tf.estimator.ModeKeys.PREDICT == mode:
                labels = features

            total_loss, length_predictions, avg_hook, prediction = self.assembler.assemble_model(features, labels, is_training)
            self.assembler.log_model_info()

            show_layer_outputs()

            if mode == tf.estimator.ModeKeys.TRAIN:
                tvars = list(tf.compat.v1.trainable_variables())
                regvars = [var for var in tvars if "CvK" in var.name or "kernel" in var.name]

                optimizer = RAdamOptimizer(config.learning_rate, L2_decay=config.L2_decay, clip_gradients=True,
                                           decay_vars=regvars, total_steps=config.train_steps)

                if config.mixed_precision_training:
                    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

                # Display gradients in TensorFlow

                if config.log_gradients:
                    grads = optimizer.compute_gradients(total_loss)
                    for g in grads:
                        tf.summary.histogram(f"{g[1].name}-grad", g[0], family="gradients")

                train_op = optimizer.minimize(
                    loss=total_loss,
                    global_step=tf.compat.v1.train.get_global_step()
                )

                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[summary_hook("train")] if config.enable_summary_hooks else []
                )

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    loss=total_loss,
                    evaluation_hooks=[summary_hook("eval"), avg_hook] if config.enable_summary_hooks else []
                )

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.PREDICT,
                    loss=total_loss,
                    predictions=prediction
                )

        return model_fn
