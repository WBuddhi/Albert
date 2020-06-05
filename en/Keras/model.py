import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from typing import Dict


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

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
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


class StsSiameseModel(tf.keras.Model):
    def __init__(self, albert_hub_model: str):
        """
        ALBERT STS-B Siamese Model

        Args:
            albert_hub_model (str): albert model tf hub path.
        """

        super().__init__()
        self.albert_hub_model = albert_hub_model
        self.pretrained_layer_1 = hub.KerasLayer(
            self.albert_hub_model, trainable=True, name="albert_layer_1",
        )
        self.pretrained_layer_2 = hub.KerasLayer(
            self.albert_hub_model, trainable=True, name="albert_layer_2",
        )
        #self.dropout_1 = tf.keras.layers.Dropout(
        #    rate=0.1, name="dropout_layer_1"
        #)
        #self.dropout_2 = tf.keras.layers.Dropout(
        #    rate=0.1, name="dropout_layer_2"
        #)
        self.cosine_layer = tf.keras.layers.Dot(
            axes=1, normalize=True, name="cosine_layer", trainable=False,
        )

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Keras Model call fn."""

        # TODO: create loss function with cosine sim and remove dot layer from model
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
        siamese_1_output, _ = self.pretrained_layer_1(inputs_text_a)
        siamese_2_output, _ = self.pretrained_layer_2(inputs_text_b)

        #if training:
        #    siamese_1_output = self.dropout_1(
        #        siamese_1_output, training=training
        #    )

        #    siamese_2_output = self.dropout_2(
        #        siamese_2_output, training=training
        #    )

        output = self.cosine_layer([siamese_1_output, siamese_2_output])
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
            "text_a": {
                "input_word_ids": sample_tensor,
                "input_mask": sample_tensor,
                "segment_ids": sample_tensor,
            },
            "text_b": {
                "input_word_ids": sample_tensor,
                "input_mask": sample_tensor,
                "segment_ids": sample_tensor,
            },
        }
        return inputs
