"""
Model training util functions.
"""
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1.keras.backend as K


def run_test(
    model: keras.Model, test_dataset: tf.data.Dataset, result_file: str
):
    """
    Run tests.

    Args:
        model (keras.Model): model
        test_dataset (tf.data.Dataset): test_dataset
        result_file (str): result_file
    """
    predictions = model.predict(x=test_dataset)
    output_data = [{"prediction": pred} for pred in predictions]
    df = pd.DataFrame(output_data)
    df.to_csv(result_file, index=False)
    tf.logging.info(f"Results saved at: {result_file}")


def print_summary(model: keras.Model, sequence_len):
    """
    Print given models summary and save to file.

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
