import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K


class StsbHead(tf.keras.layers.Layer):
    def __init__(self, name: str = "stsb_head"):
        """
        STS-B Task custom head.

        Args:
            name: layer name.
        """

        super(StsbHead, self).__init__(name=name)
        self.name = name

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

        output_dropout = self.dropout(inputs, training=training)
        output_dense = self.dense(output_dropout)
        output = tf.squeeze(output_dense, [-1])
        return output


class StsbModel(tf.keras.Model):
    def __init__(self, albert_hub_model: str, name:str="stsb_model"):
        """
        ALBERT STS-B model.

        Args:
            albert_hub_model (str): albert model tf hub path.
            name (str): name
        """
        super(StsbModel, self).__init__()
        self.name = name
        self.albert_hub_model = albert_hub_model
        self.pretrained_layer = hub.KerasLayer(
            self.albert_hub_model, trainable=True,
        )
        self.custom_head = StsbHead()

    def call(self, inputs):
        """
        Keras Model call fn.

        Args:
            inputs: Model inputs.
        return:
            Model predictions
        """
        predictions = self.custom_head(self.pretrained_layer(inputs))
        return predictions

    def get_config(self):
        """Update config."""
        config = super(StsbModel, self).get_config()
        config.update({"albert_hub_model": self.albert_hub_model})
