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
import tensorflow_hub as hub
from model import StsSiameseModel
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

        # Model Subclass
        # model = StsSiameseModel(config.get("albert_hub_module_handle", None))
        # print_summary(model, seq_len)

        # Model API
        model = create_siamese_model(
            config.get("albert_hub_module_handle", None), seq_len
        )
        model.summary()

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


def create_albert(albert_model_hub, seq_len):

    albert_inputs = [
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids"),
    ]
    pretrained_layer = hub.KerasLayer(
        albert_model_hub, trainable=True, name="albert_layer",
    )
    pooled_output, seq_output = pretrained_layer(albert_inputs)
    albert_model = keras.Model(albert_inputs, pooled_output, name="albert")
    return albert_model

def create_albert_pooled(albert_model_hub, seq_len):

    albert_inputs = [
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids"),
    ]
    pretrained_layer = hub.KerasLayer(
        albert_model_hub, trainable=True, name="albert_layer",
    )
    pooling_layer = keras.layers.GlobalAveragePooling1D(name="pooling_layer")

    pooled_output, seq_output = pretrained_layer(albert_inputs)
    output = pooling_layer(seq_output)
    albert_model = keras.Model(albert_inputs, output, name="albert")
    return albert_model

def create_siamese_model(albert_model_hub, seq_len):

    albert_model = create_albert_pooled(albert_model_hub, seq_len)
    inputs = {
        "text_a": {
            "input_word_ids": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="input_word_ids_a"
            ),
            "input_mask": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="input_mask_a"
            ),
            "segment_ids": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="segment_ids_a"
            ),
        },
        "text_b": {
            "input_word_ids": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="input_word_ids_b"
            ),
            "input_mask": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="input_mask_b"
            ),
            "segment_ids": keras.Input(
                shape=(seq_len,), dtype=tf.int32, name="segment_ids_b"
            ),
        },
    }
    inputs_text_a = [
        inputs["text_a"]["input_word_ids"],
        inputs["text_a"]["input_mask"],
        inputs["text_a"]["segment_ids"],
    ]
    inputs_text_b = [
        inputs["text_b"]["input_word_ids"],
        inputs["text_b"]["input_mask"],
        inputs["text_b"]["segment_ids"],
    ]
    cosine_layer = tf.keras.layers.Dot(
        axes=1, normalize=True, name="cosine_layer",
    )
    dropout_layer = tf.keras.layers.Dropout(rate=0.1, name="dropout")

    siamese_1_output_pooled = albert_model(inputs_text_a)
    siamese_2_output_pooled = albert_model(inputs_text_b)

    siamese_1_output_pooled = dropout_layer(siamese_1_output_pooled)
    siamese_2_output_pooled = dropout_layer(siamese_2_output_pooled)

    output = cosine_layer([siamese_1_output_pooled, siamese_2_output_pooled])
    model = keras.Model(inputs, output)
    return model


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
