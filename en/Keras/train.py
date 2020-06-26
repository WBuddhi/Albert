from typing import Tuple, List
import os
import argparse
from datetime import datetime
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1 import keras

from transformers import TFAutoModel
#from models.sts import StsSiameseModel
from models.utils import run_test, print_summary, pearson_correlation_metric_fn
from preprocess import generate_example_datasets
from optimizer.create_optimizers import (
    create_adam_decoupled_optimizer_with_warmup,
)
from utils import read_yaml_config, import_fn
from scipy.stats import pearsonr

def get_training_strategy(use_tpu:bool, tpu_name:str):

    if use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)
    else:
        return tf.distribute.MirroredStrategy()

def train_model(config: dict):
    """
    Model training function.

    Args:
        config (dict): config
    """

    tf.enable_eager_execution()
    logging.set_verbosity(tf.logging.DEBUG)
    logging.propagate = False

    use_tpu = config.get("use_tpu", False)
    tpu_name = config.get("tpu_name", None)

    model_class = config.get("model", None)
    model_name_path = config.get("transformer_name_path", None)
    pretrained_model_name = config.get("pretrained_model_name", "Unknown")
    use_avg_pooled_model = config.get("use_pretrain_avg_pooling", True)
    use_dropout = config.get("use_dropout", True)
    num_of_epochs = config.get("num_train_epochs", 5)
    log_dir = config.get("tensorboard_logs", None) + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    seq_len = config.get("sequence_len", 512)

    do_training = config.get("do_train", True)
    do_test = config.get("do_test", False)


    strategy = get_training_strategy(use_tpu, tpu_name)
    model_class = import_fn("models.sts", model_class)

    (model_datasets, config,) = generate_example_datasets(config)
    training_size = config.get("train_size", None)
    train_batch_size = config.get("train_batch_size", None)

    # TPU init code
    with strategy.scope():
        model = model_class(
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

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0
    )
    if do_training:
        model.fit(
            x=model_datasets["train"],
            epochs=num_of_epochs,
            steps_per_epoch=int(training_size/train_batch_size),
            validation_data=model_datasets["eval"],
            callbacks=[tensorboard_callback],
        )
    if do_test:
        run_test(model, model_datasets["test"])


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
