import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K


class AlbertLayer(tf.keras.layers.Layer):
    def __init__(self, config: dict, train_layers: bool = True, **kwargs):
        """
        Albert Model converted to keras layer.

        Args:
            config: configuration file
            train_layers: allow albert variables to be trained
        """

        self.trainable = train_layers
        self.config = config
        super(AlbertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""

        tf.logging.debug(input_shape)
        self.albert = hub.Module(
            self.config["albert_hub_module_handle"],
            trainable=self.trainable,
            name="{}_module".format(self.name),
        )
        albert_vars = self.albert.variables
        if self.trainable:
            self._trainable_weights.extend(
                [var for var in albert_vars if "/cls/" not in var.name]
            )
        super(AlbertLayer, self).build(input_shape)

    def call(self, inputs: list) -> tf.Tensor:
        """
        Layer call function.

        Args:
            inputs: inputs to layer [input_id, input_mask, input_segment]
        return:
            result: output tensor.
        """

        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        albert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.albert(
            inputs=albert_inputs, signature="tokens", as_dict=True
        )["pooled_output"]

        return result


class StsbHead(tf.keras.layers.Layer):
    def __init__(self, layer_size: int, name: str = "STS-B Head"):
        """
        STS-B Task custom head.

        Args:
            layer_size: unit size of layers
            name: layer name.
        """

        super(StsbHead, self).__init__(name=name)

        self.dropout = tf.keras.layers.Dropout(rate=0.1, name="dropout")

        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        bias_initializer = tf.keras.initializers.zeros()
        self.dense = tf.keras.layers.Dense(
            units=layer_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="output_weights",
        )

    def call(self, inputs: tf.Tensor, training: bool = False):
        """
        Keras Layer call function.

        Args:
            inputs: layer input, pretrained model output
            training: training model True/False
        """

        output_dropout = self.dropout(inputs, training=training)
        predictions = self.dense(output_dropout)
        return predictions


class StsbModel(tf.keras.Model):
    def __init__(self, config: dict, pretrain_train_mode: bool = True):
        """
        ALBERT STS-B model.

        Args:
            pretrain_train_mode: train layers in pretrain model
        """
        super(StsbModel, self).__init__()
        self.pretrained_layer = AlbertLayer(
            config, train_layers=pretrain_train_mode
        )
        hidden_size = 768
        self.custom_head = StsbHead(hidden_size)

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
