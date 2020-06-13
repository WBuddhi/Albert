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

        model = create_siamese_model(
            config.get("transformer_name_path", None),
            seq_len,
            use_dropout=config.get("use_dropout", True),
            pretrained_model_name=config.get(
                "pretrained_model_name", "Albert"
            ),
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


def create_albert_cls(
    model_name_path: str, seq_len: int, name: str = "Albert"
) -> keras.Model:
    """
    Create pretrained model based on classification output.

    Model uses pooled output from model which uses the classification
    embedding to represent sentence embedding.

    Args:
        model_name_path (str): model_name_path
        seq_len (int): seq_len
        name (str): name

    Returns:
        keras.Model:
    """

    albert_inputs = [
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids"),
    ]
    pretrained_layer = TFAutoModel.from_pretrained(model_name_path)
    seq_output, pooled_output = pretrained_layer(albert_inputs)
    model = keras.Model(albert_inputs, pooled_output, name="Albert")
    return model


def create_pretrained_pooled_model(
    model_name_path: str, seq_len: int, name: str = "Albert"
) -> keras.Model:
    """
    Create pretrained model based on pooled sequence embedding.

    Model uses sequence embeddings which is masked using attention mask
    input into the model and applying mean pooling to create sentence
    embedding.

    Args:
        model_name_path (str): model_name_path
        seq_len (int): seq_len
        name (str): name

    Returns:
        keras.Model:
    """

    inputs = [
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="input_ids"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="attention_mask"),
        keras.Input(shape=(seq_len,), dtype=tf.int32, name="token_type_ids"),
    ]
    pretrained_layer = TFAutoModel.from_pretrained(model_name_path)
    seq_output, pooled_output = pretrained_layer(inputs)
    tf.logging.debug(pooled_output)

    avg_masked_pooling_layer = keras.layers.GlobalAveragePooling1D(
        name="Avg_masked_pooling_layer"
    )

    seq_output, pooled_output = pretrained_layer(inputs)
    attention_mask = inputs[1]
    output = avg_masked_pooling_layer(inputs=seq_output, mask=attention_mask)

    model = keras.Model(inputs, output, name=name)
    return model


def create_siamese_model(
    model_name_path: str,
    seq_len: int,
    use_dropout: bool = True,
    pretrained_model_name: str = "Albert",
) -> keras.Model:
    """
    Create siamese model on defined Pretrained model.

    Args:
        model_name_path (str): model_name_path
        seq_len (int): seq_len
        use_dropout (bool): use_dropout
        pretrained_model_name (str): pretrained_model_name

    Returns:
        keras.Model:
    """

    pretrained_model = create_pretrained_pooled_model(
        model_name_path, seq_len, pretrained_model_name
    )
    cosine_layer = tf.keras.layers.Dot(
        axes=1, normalize=True, name="cosine_layer",
    )

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
    siamese_1_output_pooled = pretrained_model(inputs_text_a)
    siamese_2_output_pooled = pretrained_model(inputs_text_b)

    if use_dropout:
        dropout_layer = tf.keras.layers.Dropout(rate=0.1, name="dropout")
        siamese_1_output_pooled = dropout_layer(siamese_1_output_pooled)
        siamese_2_output_pooled = dropout_layer(siamese_2_output_pooled)

    output = cosine_layer([siamese_1_output_pooled, siamese_2_output_pooled])
    model = keras.Model(inputs, output)
    return model


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
