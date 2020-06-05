import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from typing import Dict


class StsbHead(tf.keras.layers.Layer):
    def __init__(self, name: str = "stsb_head"):
        """
        STS-B Task custom head.

        Args:
            name: layer name.
        """

        super(StsbHead, self).__init__(name=name)

        self.dropout = tf.keras.layers.Dropout(rate=0.1, name="dropout_layer")

        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        bias_initializer = tf.keras.initializers.zeros()
        self.dense = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="sts_head_output",
        )

    def call(self, inputs: tf.Tensor, training: bool = False):
        """
        Keras Layer call function.

        Args:
            inputs: layer input, pretrained model output
            training: training model True/False
        """

        if training:
            output_dropout = self.dropout(inputs, training=training)
        else:
            output_dropout = inputs
        output_dense = self.dense(output_dropout)
        output = tf.squeeze(output_dense, [-1])
        return output


class StsbModel(tf.keras.Model):
    def __init__(self, albert_hub_model: str):
        """
        ALBERT STS-B model.

        Args:
            albert_hub_model (str): albert model tf hub path.
        """
        super(StsbModel, self).__init__()
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        bias_initializer = tf.keras.initializers.zeros()

        self.albert_hub_model = albert_hub_model
        self.pretrained_layer = hub.KerasLayer(
            self.albert_hub_model, trainable=True, name="albert_layer",
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.1, name="dropout_layer")
        self.dense = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="sts_head_output",
        )

    def call(self, inputs:tf.Tensor, training:bool=None) -> tf.Tensor:
        """
        Keras Model call fn.
        """
        inputs = [
            keras.Input(
                dtype=tf.int32,
                name="input_word_ids",
                tensor=inputs["input_word_ids"],
            ),
            keras.Input(
                dtype=tf.int32, name="input_mask", tensor=inputs["input_mask"],
            ),
            keras.Input(
                dtype=tf.int32,
                name="segment_ids",
                tensor=inputs["segment_ids"],
            ),
        ]
        output, _ = self.pretrained_layer(inputs)
        output = self.dropout(output, training=training)
        output = self.dense(output)
        output = tf.squeeze(output, [-1], name="output")
        return output

    def get_config(self) -> Dict[str, str]:
        """Update config."""
        config = super(StsbModel, self).get_config()
        config.update({"albert_hub_model": self.albert_hub_model})
        return config

    def get_sample_input(self, sequence_len: int) -> Dict[str, keras.Input]:
        sample_tensor = keras.Input(shape=(sequence_len,), dtype=tf.int32)
        inputs = {
            "input_word_ids": sample_tensor,
            "input_mask": sample_tensor,
            "segment_ids": sample_tensor,
        }
        return inputs
