from typing import Tuple
import os
import tensorflow.compat.v1 as tf

# import tensorflow as tf
from tensorflow.compat.v1 import logging
import tensorflow.compat.v1.keras.backend as K
from model import StsbModel, AlbertLayer, StsbHead
from dataprocessor import DataProcessor, StsbProcessor
from preprocess import (
    file_based_input_fn_builder,
    file_based_convert_examples_to_features,
)
from optimizer.polynomial_decay_with_warmup import PolynomialDecayWarmup
from optimizer.adamw import AdamWeightDecayOptimizer
from utils import read_yaml_config
import argparse
from tokenization import FullTokenizer
import tensorflow_hub as hub
from datetime import datetime
from tensorflow.compat.v1 import keras


def train_model(config: dict):
    """
    Model training function.

    Args:
        config (dict): config
    """

    tf.enable_eager_execution()
    logging.set_verbosity(tf.logging.DEBUG)
    logging.propagate = False
    stsb_processor = StsbProcessor(
        config["spm_model_file"], config["do_lower_case"]
    )
    train_examples = stsb_processor.get_train_examples(config["data_dir"])
    config["training_steps"] = len(train_examples)
    # TPU init code
    if config.get("use_tpu", False):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=config["tpu_name"]
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    metrics = [
        keras.metrics.mean_squared_error
        #        pearson_correlation_metric_fn,
    ]
    seq_len = config.get("sequence_len", 512)
    with strategy.scope():
        inputs = {}
        inputs["input_word_ids"] = keras.Input(
            shape=(seq_len,), dtype=tf.int32, name="input_word_ids"
        )
        inputs["input_mask"] = keras.Input(
            shape=(seq_len,), dtype=tf.int32, name="input_mask"
        )
        inputs["segment_ids"] = keras.Input(
            shape=(seq_len,), dtype=tf.int32, name="segment_ids"
        )
        albert_layer = hub.KerasLayer(
            config.get("albert_hub_module_handle", ""), trainable=True,
        )
        # albert_trainable_vars = [
        #    var for var in albert_layer.variables if "/cls/" not in var.name
        # ]
        # albert_layer._trainable_weights.extend(albert_trainable_vars)
        # for var in albert_trainable_vars:
        #    albert_layer._non_trainable_weights.remove(var)
        #    pass
        dropout_layer = keras.layers.Dropout(rate=0.1, name="dropout_layer")
        kernel_init = keras.initializers.TruncatedNormal(stddev=0.02)
        bias_init = keras.initializers.zeros()
        dense_layer = keras.layers.Dense(
            units=1,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name="dense_layer",
        )
        albert_pooled_output, _ = albert_layer(list(inputs.values()))
        logging.debug(albert_pooled_output)
        dropout = dropout_layer(albert_pooled_output)
        output = dense_layer(dropout)
        output = tf.squeeze(output, [-1])
        model = keras.Model(inputs, output)
        mse_loss = keras.losses.MeanSquaredError()
        mse_metrics = keras.metrics.MeanSquaredError(dtype=tf.float32)

        logging.debug(model.summary())
        optimizer = _create_optimizer(config)
    model.compile(
        optimizer=optimizer, loss=mse_loss, metrics=[mse_metrics],
    )

    train_file, eval_file, test_file = _create_train_eval_input_files(
        config, stsb_processor
    )
    train_dataset = file_based_input_fn_builder(
        train_file, seq_len, is_training=True,
    )
    eval_dataset = file_based_input_fn_builder(
        eval_file, seq_len, is_training=False,
    )
    test_dataset = file_based_input_fn_builder(
        test_file, seq_len, is_training=False,
    )
    log_dir = "gs://buddhi_albert/model_logs/" + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0
    )
    model.fit(
        x=train_dataset,
        epochs=config.get("num_train_epochs", 5),
        steps_per_epoch=int(
            len(stsb_processor.get_train_examples(config["data_dir"])) / 32
        ),
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback],
    )


def _create_train_eval_input_files(
    config: dict, processor: DataProcessor
) -> Tuple[str]:
    """
    Create training and eval input files.

    Args:
        config (dict): config
        processor (DataProcessor): processor

    Returns:
        Tuple[str]: (training data file,
            evaluation data file,
            test data file)
    """
    cached_dir = config.get("cached_dir", None)
    task_name = config.get("task_name", "Experiment")
    data_dir = config.get("data_dir", "")
    if not cached_dir:
        cached_dir = config.get("output_dir", None)
    train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
    train_examples = processor.get_train_examples(data_dir)
    eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
    eval_examples = processor.get_dev_examples(data_dir)
    test_file = os.path.join(cached_dir, task_name + "_test.tf_record")
    test_examples = processor.get_test_examples(data_dir)
    label_list = processor.get_labels()
    tokenizer = _get_tokenizer(config)
    for data_file, examples in zip(
        (train_file, eval_file, test_file),
        (train_examples, eval_examples, test_examples),
    ):
        file_based_convert_examples_to_features(
            examples,
            label_list,
            config.get("sequence_len", 512),
            tokenizer,
            data_file,
            task_name,
        )
    return train_file, eval_file, test_file


def _get_tokenizer(config: dict) -> FullTokenizer:
    """
    Get tokenizer.

    Args:
        config (dict): config

    Returns:
        FullTokenizer:
    """
    return FullTokenizer(
        vocab_file=None,
        do_lower_case=config.get("do_lower_case", True),
        spm_model_file=config.get("spm_model_file", ""),
    )


# def pearson_correlation_metric_fn(
#    y_true: tf.Tensor, y_pred: tf.Tensor
# ) -> tf.contrib.metrics:
#    """
#    Pearson correlation metric function.
#
#    Args:
#        y_true (tf.Tensor): y_true
#        y_pred (tf.Tensor): y_pred
#
#    Returns:
#        tf.contrib.metrics: pearson correlation
#    """
#    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]


def _create_optimizer(config: dict) -> AdamWeightDecayOptimizer:
    """
    Create optimizer.

    Args:
        config (dict): config

    Returns:
        AdamWeightDecayOptimizer:
    """
    logging.debug("Creating optimizer.")
    init_lr = float(config.get("learning_rate", 5e-5))
    batch_size = config.get("train_batch_size", 32)
    train_epochs = config.get("num_train_epochs", 5)
    num_warmup_steps = config.get("warmup_steps", 0)
    weight_decay_rate = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-6
    exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"]
    training_len = config.get("training_steps", 5000)
    num_train_steps = int((training_len / batch_size) * train_epochs)
    params_log = {
        "Initial learning rate": init_lr,
        "Number of training steps": num_train_steps,
        "Number of warmup steps": num_warmup_steps,
        "End learning rate": 0.0,
        "Weight decay rate": weight_decay_rate,
        "Beta_1": beta_1,
        "Beta_2": beta_2,
        "Epsilon": epsilon,
        "Excluded layers from weight decay": exclude_from_weight_decay,
    }
    logging.debug("Optimizer Parameters")
    logging.debug("=" * 20)
    for key, value in params_log.items():
        logging.debug(f"{key}: {value}")
    learning_rate = PolynomialDecayWarmup(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        end_learning_rate=0.0,
    )

    keras.utils.get_custom_objects()[
        "PolynomialDecayWarmup"
    ] = PolynomialDecayWarmup
    return AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=exclude_from_weight_decay,
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
