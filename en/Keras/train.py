from typing import Tuple
import os
import argparse
from datetime import datetime
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1 import keras
from models.sts import StsbModel
from models.utils import run_test, print_summary, pearson_correlation_metric_fn
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
    (model_datasets, config,) = generate_example_datasets(config)
    # TPU init code
    with strategy.scope():

        model_name_path = config.get("transformer_name_path", None)
        pretrained_model_name = config.get("pretrained_model_name", "Unknown")
        use_avg_pooled_model = config.get("use_pretrain_avg_pooling", True)
        use_dropout = config.get("use_dropout", True)

        model = StsbModel(
            model_name_path,
            seq_len,
            pretrained_model_name,
            use_avg_pooled_model,
            use_dropout,
        )
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
        x=model_datasets["train"],
        epochs=config.get("num_train_epochs", 5),
        steps_per_epoch=int(
            config.get("train_size", None)
            / config.get("train_batch_size", None)
        ),
        validation_data=model_datasets["eval"],
        callbacks=[tensorboard_callback],
    )
    if config.get("do_test", False):
        run_test(
            model,
            model_datasets["test"],
            config.get("pred_file", "results.csv"),
        )


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
