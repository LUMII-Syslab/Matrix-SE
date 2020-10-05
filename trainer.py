#!/usr/bin/python3
import argparse
import itertools
import os
import shutil
import time
from pathlib import Path

import tensorflow as tf

import config
from data.dataset_builder import DatasetBuilder
from models.model_builder import GPUModelBuilder


def clear_all():
    print("Clearing old data/checkpoints!")
    if Path(config.train_dir).exists():
        shutil.rmtree(config.train_dir)


def train(model, dataset_builder):
    tf.compat.v1.logging.info("Training model on train data!")
    train_params = {"batch_size": config.train_batch_size}
    eval_params = {"batch_size": config.eval_batch_size}

    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset_builder.make_train_fn(train_params),
        max_steps=config.train_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=dataset_builder.make_eval_fn(eval_params),
        steps=config.eval_steps,
        throttle_secs=300
    )

    tf.estimator.train_and_evaluate(
        model,
        train_spec,
        eval_spec
    )


def evaluate(model, dataset_builder):
    tf.compat.v1.logging.info("Evaluating model on test data!")
    eval_params = {"batch_size": config.eval_batch_size}
    eval_results = model.evaluate(
        input_fn=dataset_builder.make_eval_fn(eval_params),
        steps=config.eval_steps
    )

    print(eval_results)


def create_model():
    model_builder = GPUModelBuilder(config.model, config.dataset)

    session_config = tf.compat.v1.ConfigProto()
    optimizer_options = session_config.graph_options.optimizer_options
    if config.use_xla:
        optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        save_summary_steps=None,
        save_checkpoints_steps=config.save_checkpoint_steps,
        session_config=session_config,
        keep_checkpoint_max=10
    )

    model = tf.estimator.Estimator(
        model_fn=model_builder.create_model_fn(),
        config=run_config
    )

    return model


def test(model, dataset_builder):
    tf.compat.v1.logging.info("Evaluating model on whole test data!")
    test_params = {"batch_size": config.test_batch_size}
    input_fn = dataset_builder.make_test_fn(test_params)
    eval_results = model.evaluate(input_fn=input_fn)
    print(eval_results)


def main():
    if args.clear:
        clear_all()

    if args.label:
        config.model_dir = config.model_dir + "_" + args.label

    if args.model_dir:
        config.model_dir = args.model_dir

    if args.time:
        config.eval_batch_size = 1
        config.shuffle_size = 1
        config.eval_steps = 1
        try:
            measure_time()
        except BaseException as exp:
            print("Time measuring has successfully failed!")
            raise exp

    if args.train or args.eval or args.test:
        dataset_builder = DatasetBuilder(config.dataset)
        model = create_model()

        if args.train:
            train(model, dataset_builder)

        if args.eval:
            evaluate(model, dataset_builder)

        if args.test:
            test(model, dataset_builder)

    if args.gen:
        config.eval_batch_size = 1
        config.shuffle_size = 1
        config.eval_steps = 500  # generate less examples
        try:
            check_generalization()
        except BaseException as exp:
            print("Great success! Generalization test complete!")
            raise exp


def check_generalization():
    length = min(config.dataset.config["train_lengths"])
    while length <= 1024:
        # start of black magic
        dataset = type(config.dataset)
        dataset.eval_size = property(lambda self: config.eval_steps)  # use smaller test set
        dataset = dataset()
        dataset.add_config("eval_lengths", [length])  # eval on single length at the time
        # end of black magic

        dataset_builder = DatasetBuilder(dataset)
        model = create_model()
        evaluate(model, dataset_builder)
        length *= 2


def measure_time():
    tf.compat.v1.logging.info("Measuring prediction time!")

    model = create_model()
    length = 4
    summary = "Summary on time measurements\n"

    while length <= 1024:
        # start of black magic
        dataset = type(config.dataset)
        dataset.eval_size = property(lambda self: config.eval_steps)  # use smaller test set
        dataset = dataset()
        dataset.add_config("eval_lengths", [length])  # eval on single length at the time
        # end of black magic

        dataset_builder = DatasetBuilder(dataset)
        tf.compat.v1.logging.info(f"Measuring prediction time on length {length}")
        eval_params = {"batch_size": config.eval_batch_size}

        results_gen = model.predict(
            input_fn=dataset_builder.make_eval_fn(eval_params),
            yield_single_examples=True
        )

        for i, x in enumerate(itertools.islice(results_gen, 100)):
            print(f"Discarding element {i}")

        total_time = 0
        steps = 100
        for i in range(steps):
            start_time = time.time()
            next(results_gen)
            total_time += time.time() - start_time
        summary += f"Average prediction time on {length}x{length} matrix over {steps} steps: {total_time/steps}s\n"

        length *= 2

    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="evaluate model on test data")
    parser.add_argument("--continue", action="store", dest="model_dir", help="continue training the model", type=str)
    parser.add_argument("--label", action="store", help="add label to model_dir", type=str)
    parser.add_argument("--train", action="store_true", help="train model on train data")
    parser.add_argument("--clear", action="store_true", help="clear checkpoints and cache")
    parser.add_argument("--gen", action="store_true", help="Test generalization")
    parser.add_argument("--test", action="store_true", help="Test on whole test or eval dataset")
    parser.add_argument("--time", action="store_true", help="Measure average prediction time for one element")
    args = parser.parse_args()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_gpus
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

    main()
