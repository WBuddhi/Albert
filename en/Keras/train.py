from typing import Tuple
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import logging
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1 import keras
from model import StsbModel
from preprocess import generate_example_datasets
from optimizer.create_optimizers import (
    create_adam_decoupled_optimizer_with_warmup,
)
from utils import read_yaml_config
from scipy.stats import pearsonr


def train_model(config: dict):
    """
    Model training function.

    Args:
        config (dict): config
    """

    tf.enable_eager_execution()
    logging.set_verbosity(tf.logging.DEBUG)
    logging.propagate = False
    if config.get("use_tpu", False):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=config["tpu_name"]
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    seq_len = config.get("sequence_len", 512)
    (
        train_dataset,
        eval_dataset,
        test_dataset,
        config,
    ) = generate_example_datasets(config)
    # TPU init code
    with strategy.scope():
        model = StsbModel(config.get("albert_hub_module_handle", None))
        print_summary(model, seq_len)
        mse_loss = keras.losses.MeanSquaredError()
        optimizer = create_adam_decoupled_optimizer_with_warmup(config)
        metrics = [
            keras.metrics.MeanSquaredError(dtype=tf.float32),
            pearson_correlation_metric_fn,
        ]
        model.compile(
            optimizer=optimizer, loss=mse_loss, metrics=metrics,
        )

    log_dir = config.get("tensorboard_logs", None) + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0
    )
    model.fit(
        x=train_dataset,
        epochs=config.get("num_train_epochs", 5),
        steps_per_epoch=int(
            config.get("train_size", None)
            / config.get("train_batch_size", None)
        ),
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback],
    )
    if config.get("do_predict", False):
        run_test(model)


def run_test(
    model: keras.Model, test_dataset: tf.data.Dataset,
):
    predictions = model.predict(x=test_dataset)
    output_data = [{"prediction": pred} for pred in predictions]
    df = pd.DataFrame(output_data)
    result_file = config.get("pred_file", "results.csv")
    df.to_csv(result_file, index=False)
    tf.logging.info(f"Results saved at: {result_file}")


def print_summary(model: keras.Model, sequence_len):
    sample_input = model.get_sample_input(sequence_len)
    model(sample_input)
    model.summary()


def pearson_correlation_metric_fn(
    y_true: tf.Tensor, y_pred: tf.Tensor,
) -> tf.Tensor:
    """
    Pearson correlation metric function.
    https://github.com/WenYanger/Keras_Metrics

    Args:
        y_true (tf.Tensor): y_true
        y_pred (tf.Tensor): y_pred

    Returns:
        tf.contrib.metrics: pearson correlation
    """

    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum) + 1e-12
    r = r_num / r_den
    return K.mean(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="config file path",
        default="./config.yaml",
        type=str,
    )
    args = parser.parse_args()
    config = read_yaml_config(args.config_file)
    train_model(config)
