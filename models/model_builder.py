from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data.base import Dataset
from models.base import Model
from optimization.radam import RAdamOptimizer
from utils.log_memory import show_layer_outputs


def colorize(value: tf.Tensor, colors: tf.Tensor):
    """ Convert IDs to color image """
    dim = value.shape.as_list()
    value = tf.reshape(value, [dim[0], dim[1] * dim[2], 1])
    value = tf.gather(colors, value)

    value = tf.cast(value, dtype=tf.uint8)
    value = tf.reshape(value, [dim[0], dim[1], dim[2], 3])
    return value


def color_map():
    return tf.constant([
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ])


def chess_color_map():
    return tf.constant([
        [0, 0, 0],  # Empty space
        [153, 255, 153],  # Q green
        [153, 153, 255],  # R blue
        [255, 153, 255],  # B purple
        [255, 255, 153],  # N yellow
        [255, 255, 255],  # P white
        [204, 0, 0],  # k red
        [0, 204, 0],  # q green
        [0, 0, 204],  # r blue
        [153, 0, 153],  # b purple
        [153, 153, 0],  # n yellow
        [160, 160, 160],  # p gray
        [255, 153, 153]  # K red
    ])


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
    """ Calculates average value over steps"""

    def __init__(self, tensors: dict, file_name=None):
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
            with open(self._file_name, "a+") as file:
                self.__log(file.write)
                self.__log(print)
        else:
            self.__log(print)

    def __log(self, output_stream):
        for key, value in self._total_value.items():
            output_stream(f"Average {key} over {self._steps} steps: {value / self._steps}\n")


class ModelAssembler:
    """
    Initiates model for each input size in datasets. All models share the same weights.
    """
    def __init__(self, model: Model, dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset

    def assemble_model(self, features, labels, training: bool):
        input_classes = self.dataset.config["input_classes"]
        output_classes = self.dataset.config["output_classes"]

        self.model.build(input_classes, output_classes)

        total_loss = 0.0
        length_predictions = []

        # colors = color_map()
        log_dict = {}

        for inputs, targets in zip(features, labels):
            length = inputs.shape.as_list()[1]
            logits = self.model.call(inputs, training=training)

            predictions = {
                'classes': tf.argmax(input=logits, axis=-1, output_type=tf.int32),
                'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
            }

            if config.chess:
                # flatten
                logits = tf.reshape(logits, [logits.shape[0], -1])
                # dense 64x64
                logits = tf.keras.layers.Dense(units=64 * 64)(logits)
                new_targets = tf.reshape(targets, [targets.shape[0], 64])
                new_targets = tf.one_hot(new_targets, depth=3, dtype=tf.int32)
                new_targets = tf.argmax(new_targets, axis=1)
                new_targets = tf.gather(new_targets, [1, 2], axis=1)
                new_targets = tf.reduce_sum(new_targets * [64, 1], axis=1)
                output_classes = 64 * 64
                loss = self.calculate_loss_with_smoothing(new_targets, logits, output_classes,
                                                          config.adjust_class_imbalance)
            else:
                loss = self.calculate_loss_with_smoothing(targets, logits, output_classes,
                                                          config.adjust_class_imbalance)

            if config.use_jaccard_loss:
                jac = self.jaccard_distance(targets, predictions["probabilities"], output_classes)
                tf.compat.v1.summary.scalar(f"jaccard_loss_{length}", loss, family="loss")
                loss += jac

            # flooding_level = 0.001
            # loss = tf.math.abs(loss - flooding_level) + flooding_level

            total_loss += loss

            tf.compat.v1.summary.scalar(f"loss_{length}", loss, family="loss")

            if config.chess:
                predicted_classes = tf.nn.softmax(logits)
                predicted_classes = tf.argmax(predicted_classes, axis=1)
                div_tensor = tf.floor_div(predicted_classes, [64])
                mod_tensor = tf.floormod(predicted_classes, [64])
                predicted_classes = tf.stack([div_tensor, mod_tensor], axis=1)
                predicted_classes = tf.one_hot(predicted_classes, depth=64, dtype=tf.int32) * tf.stack(
                    [tf.fill([64], 1), tf.fill([64], 2)])
                predicted_classes = tf.reshape(tf.reduce_sum(predicted_classes, axis=1), [logits.shape[0], 8, 8])
                predictions["classes"] = predicted_classes
                image1 = colorize(inputs, chess_color_map())
                tf.compat.v1.summary.image("inputs", image1, max_outputs=3, family="inputs")
                image2 = colorize(targets, chess_color_map())
                tf.compat.v1.summary.image("labels", image2, max_outputs=3, family="labels")
                image3 = colorize(predictions["classes"], chess_color_map())
                tf.compat.v1.summary.image("predictions", image3, max_outputs=3, family="predictions")

            log_dict[f"accuracy_{length}x{length}"] = get_accuracy(predictions["classes"], targets)
            tf.compat.v1.summary.scalar(f"{length}x{length}", log_dict[f"accuracy_{length}x{length}"],
                                        family="accuracy")

            log_dict[f"total_accuracy_{length}x{length}"] = get_all_correct(predictions["classes"], targets)
            tf.compat.v1.summary.scalar(f"total{length}x{length}", log_dict[f"total_accuracy_{length}x{length}"],
                                        family="accuracy")

            if config.log_input_image:
                input_shape = inputs.get_shape().as_list()
                tf.compat.v1.summary.image("features_" + str(input_shape[1]), inputs, max_outputs=8)

            if config.log_segmentation_output:
                color_label = colorize(targets, color_map())
                tf.compat.v1.summary.image("labels_" + str(input_shape[1]), color_label, max_outputs=8)

                pred = colorize(predictions["classes"], color_map())
                tf.compat.v1.summary.image("predictions_" + str(input_shape[1]), pred, max_outputs=8)

                class_id_to_group = tf.constant([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7])
                group_targets = tf.gather(class_id_to_group, targets)
                group_pred = tf.gather(class_id_to_group, predictions["classes"])

                class_iou, class_iou_update = tf.metrics.mean_iou(targets, predictions["classes"], output_classes)
                group_iou, group_iou_update = tf.metrics.mean_iou(group_targets, group_pred, output_classes)

                with tf.control_dependencies([class_iou_update, group_iou_update]):
                    class_iou = tf.identity(class_iou)
                    group_iou = tf.identity(group_iou)
                    tf.compat.v1.summary.scalar(f"IoU_class_{length}", class_iou, family="accuracy")
                    tf.compat.v1.summary.scalar(f"IoU_group_{length}", group_iou, family="accuracy")

            length_predictions.append((targets, predictions["classes"]))

        tf.compat.v1.summary.scalar("total_loss", total_loss, family="loss")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            name = var.name.replace("var_lengths", "")
            tf.compat.v1.summary.histogram(name + '/histogram', var)

        gen_file = Path(config.model_dir) / config.gen_test_file
        return total_loss, length_predictions, AverageLoggingHook(log_dict, file_name=gen_file)

    @staticmethod
    def calculate_loss(labels, logits, output_classes):
        labels_one_hot = tf.one_hot(labels, output_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
        return tf.reduce_mean(loss)

    @staticmethod
    def jaccard_distance(y_true, y_pred, output_classes, smooth=1e-10):
        """ Calculatete Jaccard distance (https://en.wikipedia.org/wiki/Jaccard_index)"""
        y_true = tf.one_hot(y_true, output_classes)

        intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
        _sum = tf.reduce_sum(y_true + y_pred, axis=-1)
        jac = (intersection + smooth) / (_sum - intersection + smooth)
        return 1 - tf.reduce_mean(jac + smooth)

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
        info += ''.join(["-"] * len(header)) + "\n"

        tf.compat.v1.logging.info(info)


class ModelBuilder(metaclass=ABCMeta):

    @abstractmethod
    def create_model_fn(self):
        pass


def summary_hook(training_mode):
    output_dir = config.model_dir + "/" + training_mode
    return tf.estimator.SummarySaverHook(
        save_steps=config.save_checkpoint_steps,
        output_dir=output_dir,
        summary_op=tf.compat.v1.summary.merge_all()
    )


class GPUModelBuilder(ModelBuilder):

    def __init__(self, model: Model, dataset: Dataset) -> None:
        self.assembler = ModelAssembler(model, dataset)

    def create_model_fn(self):

        def model_fn(features: tf.data.Dataset, labels: tf.data.Dataset, mode, params):

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            total_loss, length_predictions, avg_hook = self.assembler.assemble_model(features, labels, is_training)
            self.assembler.log_model_info()

            show_layer_outputs()

            if mode == tf.estimator.ModeKeys.TRAIN:
                tvars = list(tf.compat.v1.trainable_variables())
                regvars = [var for var in tvars if "CvK" in var.name or "kernel" in var.name]

                optimizer = RAdamOptimizer(config.learning_rate, L2_decay=config.L2_decay,
                                           clip_gradients=True,
                                           decay_vars=regvars)

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

        return model_fn
