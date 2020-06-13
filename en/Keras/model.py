import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
import tensorflow_hub as hub
from transformers import TFAutoModel
from typing import Dict


class PretrainedModelAvgPooling(tf.keras.Model):
    """Avg Pooling Pretrained Model."""

    def __init__(self, model_name_path: str, name: str):
        """
        Pretrained Avg Pooling Model initializer.

        Args:
            model_name_path (str): model_name_path
            name (str): name
        """
        self.model_name_path = model_name_path
        self.name = name
        super().__init__()
        self.pretrained_layer = TFAutoModel.from_pretrained(model_name_path)
        self.avg_masked_pooling_layer = keras.layers.GlobalAveragePooling1D(
            name="Avg_masked_pooling_layer"
        )

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Keras model call fn.

        Args:
            inputs (tf.Tensor): inputs
            training (bool): training

        Returns:
            tf.Tensor:
        """
        seq_output, _ = self.pretrained_layer(inputs)
        attention_mask = inputs["attention_mask"]
        output = self.avg_masked_pooling_layer(seq_output, mask=attention_mask)
        return output

    def get_config(self) -> Dict[str, str]:
        """
        Get model config.

        Returns:
            Dict[str, str]:
        """
        config = super().get_config()
        config.update({"model_name_path": self.model_name_path})
        return config


class PretrainedModelCls(tf.keras.Model):
    """Classification Pretrained Model."""

    def __init__(self, model_name_path: str, name: str):
        """
        Pretrained Classification Model initializer.

        Args:
            model_name_path (str): model_name_path
            name (str): name
        """
        self.model_name_path = model_name_path
        self.name = name
        super().__init__()
        self.pretrained_layer = TFAutoModel.from_pretrained(model_name_path)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Keras model call fn.

        Args:
            inputs (tf.Tensor): inputs
            training (bool): training

        Returns:
            tf.Tensor:
        """

        _, cls_output = self.pretrained_layer(inputs)
        return cls_output

    def get_config(self) -> Dict[str, str]:
        """
        Get model config.

        Returns:
            Dict[str, str]:
        """
        config = super.get_config()
        config.update({"model_name_path": self.model_name_path})
        return config


class StsSiameseModel(tf.keras.Model):
    """STS-Benchmark Siamese Model."""

    def __init__(
        self,
        model_name_path: str,
        sequence_len: int,
        pretrained_model_pooled: bool,
        pretrained_model_name: str,
        use_dropout: bool = True,
        name: str = "StsSiameseModel",
    ):
        """
        STS Siamese Model initilizer.

        Args:
            model_name_path (str): model_name_path
            sequence_len (int): sequence_len
            pretrained_model_pooled (bool): pretrained_model_pooled
            pretrained_model_name (str): pretrained_model_name
            use_dropout (bool): use_dropout
            name (str): name
        """
        self.model_name_path = model_name_path
        self.sequence_len = sequence_len
        self.pretrained_model_name = pretrained_model_name
        self.use_dropout = use_dropout
        self.pretrained_model_pooled = pretrained_model_pooled
        self.name = name
        super().__init__()
        self.pretrained_model = (
            PretrainedModelAvgPooling(
                self.model_name_path, self.pretrained_model_name
            )
            if self.pretrained_model_pooled
            else PretrainedModelCls(
                self.model_name_path, self.pretrained_model_name
            )
        )
        self.dropout_layer = (
            tf.keras.Dropout(rate=0.1, name="Dropout_layer")
            if self.use_dropout
            else None
        )
        self.cosine_layer = tf.keras.Dot(
            axes=1, normalize=True, name="Cosine_layer"
        )

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = None
    ) -> tf.Tensor:
        """
        Keras Model call fn.

        Args:
            inputs (Dict[str, tf.Tensor]): inputs
            training (bool): training

        Returns:
            tf.Tensor:
        """
        siamese_1_output = self.pretrained_model(inputs["text_a"])
        siamese_2_output = self.pretrained_model(inputs["text_b"])
        if self.use_dropout and training:
            siamese_1_output = self.dropout_layer(siamese_1_output)
            siamese_2_output = self.dropout_layer(siamese_2_output)
        output = self.cosine_layer([siamese_1_output, siamese_2_output])
        return output

    def get_config(self):
        """Get Model configuration."""
        config = super.get_config()
        config.update(
            {
                "model_name_path": self.model_name_path,
                "sequence_len": self.sequence_len,
                "pretrained_model_name": self.pretrained_model_name,
                "pretrained_model_pooled": self.pretrained_model_pooled,
                "use_dropout": self.use_dropout,
            }
        )
        return config


class StsbModel(tf.keras.Model):
    def __init__(self, albert_hub_model: str):
        """
        ALBERT STS-B model.

        Args:
            albert_hub_model (str): albert model tf hub path.
        """
        super().__init__()
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

        Args:
            inputs (tf.Tensor): inputs
            training (bool): training

        Returns:
            tf.Tensor:
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
        config = super().get_config()
        config.update({"albert_hub_model": self.albert_hub_model})
        return config

    def get_sample_input(self, sequence_len: int) -> Dict[str, keras.Input]:
        """
        Generate sample inputs for model initilization.

        Args:
            sequence_len (int): sequence_len

        Returns:
            Dict[str, keras.Input]:
        """
        sample_tensor = keras.Input(shape=(sequence_len,), dtype=tf.int32)
        inputs = {
            "input_word_ids": sample_tensor,
            "input_mask": sample_tensor,
            "segment_ids": sample_tensor,
        }
        return inputs
