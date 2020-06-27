import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from transformers import TFAutoModel, TFAlbertModel
from typing import Dict


class PretrainedModelAvgPooling(tf.keras.Model):
    """Avg Pooling Pretrained Model."""

    def __init__(self, model_name_path: str, use_dropout: bool, name: str):
        """
        Pretrained Avg Pooling Model initializer.

        Args:
            model_name_path (str): model_name_path
            use_dropout (bool): use_dropout
            name (str): name
        """
        self.model_name_path = model_name_path
        self.use_dropout = use_dropout
        super().__init__()
        self.pretrained_layer = TFAutoModel.from_pretrained(model_name_path)
        self.avg_masked_pooling_layer = keras.layers.GlobalAveragePooling1D(
            name="Avg_masked_pooling_layer"
        )
        self.dropout_layer = (
            tf.keras.layers.Dropout(rate=0.1, name="Dropout_layer")
            if self.use_dropout
            else None
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
        if self.use_dropout and training:
            output = self.dropout_layer(output)
        return output

    def get_config(self) -> Dict[str, str]:
        """
        Get model config.

        Returns:
            Dict[str, str]:
        """
        config = super().get_config()
        config.update(
            {
                "model_name_path": self.model_name_path,
                "use_dropout": self.use_dropout,
            }
        )
        return config


class PretrainedModelCls(tf.keras.Model):
    """Classification Pretrained Model."""

    def __init__(self, model_name_path: str, use_dropout: bool, name: str):
        """
        Pretrained Classification Model initializer.

        Args:
            model_name_path (str): model_name_path
            use_dropout (bool): use_dropout
            name (str): name
        """
        self.model_name_path = model_name_path
        self.use_dropout = use_dropout
        super().__init__()
        self.pretrained_layer = TFAlbertModel.from_pretrained(model_name_path)
        self.dropout_layer = (
            tf.keras.layers.Dropout(rate=0.1, name="Dropout_layer")
            if self.use_dropout
            else None
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

        _, cls_output = self.pretrained_layer(inputs, training=training)
        if self.use_dropout and training:
            cls_output = self.dropout_layer(cls_output)
        return cls_output

    def get_config(self) -> Dict[str, str]:
        """
        Get model config.

        Returns:
            Dict[str, str]:
        """
        config = super.get_config()
        config.update(
            {
                "model_name_path": self.model_name_path,
                "use_dropout": self.use_dropout,
            }
        )
        return config
