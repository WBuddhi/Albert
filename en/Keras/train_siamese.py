from typing import Tuple, List
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import logging
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1 import keras

from transformers import TFAutoModel
from models.sts import StsSiameseModel
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

        model_name_path = config.get("transformer_name_path", None)
        pretrained_model_name = config.get("pretrained_model_name", "Unknown")
        use_avg_pooled_model = config.get("use_pretrain_avg_pooling", True)
        use_dropout = config.get("use_dropout", True)

        model = StsSiameseModel(
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
    if config.get("do_train", True):
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
    if config.get("do_test", False):
        run_test(model, test_dataset)


def run_test(
    model: keras.Model, test_dataset: tf.data.Dataset,
):
    """
    Run model on test set.

    Args:
        model (keras.Model): model
        test_dataset (tf.data.Dataset): test_dataset
    """
    predictions = model.predict(x=test_dataset)
    output_data = [{"prediction": pred[0]} for pred in predictions]
    df = pd.DataFrame(output_data)
    result_file = config.get("pred_file", "results.csv")
    df.to_csv(result_file, index=False)
    tf.logging.info(f"Results saved at: {result_file}")


def print_summary(model: keras.Model, sequence_len):
    """
    Prints and saves model summary.

    Args:
        model (keras.Model): model
        sequence_len:
    """
    sample_input = model.sample_input(sequence_len)
    model(sample_input)
    model.summary()
    keras.utils.plot_model(
        model,
        to_file="./model.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
    )


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
        tf.Tensor: pearson correlation
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
