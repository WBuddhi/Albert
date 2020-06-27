import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from typing import Dict
from models.pretrained import PretrainedModelCls, PretrainedModelAvgPooling


class StsSiameseModel(tf.keras.Model):
    """STS-Benchmark Siamese Model."""

    def __init__(
        self,
        model_name_path: str,
        sequence_len: int,
        pretrained_model_name: str,
        use_avg_pooled_model: bool = True,
        use_dropout: bool = True,
        name: str = "StsSiameseModel",
    ):
        """
        STS Siamese Model initilizer.

        Args:
            model_name_path (str): model_name_path
            sequence_len (int): sequence_len
            pretrained_model_name (str): pretrained_model_name
            use_avg_pooled_model (bool): use_avg_pooled_model
            use_dropout (bool): use_dropout
            name (str): name
        """
        self.model_name_path = model_name_path
        self.sequence_len = sequence_len
        self.pretrained_model_name = pretrained_model_name
        self.use_dropout = use_dropout
        self.use_avg_pooled_model = use_avg_pooled_model
        super().__init__()
        self.pretrained_model = (
            PretrainedModelAvgPooling(
                self.model_name_path, use_dropout, self.pretrained_model_name
            )
            if self.use_avg_pooled_model
            else PretrainedModelCls(
                self.model_name_path, use_dropout, self.pretrained_model_name
            )
        )
        self.cosine_layer = tf.keras.layers.Dot(
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
                "use_avg_pooled_model": self.use_avg_pooled_model,
                "use_dropout": self.use_dropout,
            }
        )
        return config

    @classmethod
    def sample_input(cls, sequence_len: int) -> Dict[str, tf.Tensor]:
        """
        Returns sample inputs.

        Args:
            sequence_len (int): sequence_len

        Returns:
            Dict[str, tf.Tensor]:
        """
        return {
            "text_a": {
                "input_ids": keras.Input(
                    shape=(sequence_len,), dtype=tf.int32, name="input_ids",
                ),
                "attention_mask": keras.Input(
                    shape=(sequence_len,),
                    dtype=tf.int32,
                    name="attention_mask",
                ),
                "token_type_ids": keras.Input(
                    shape=(sequence_len,),
                    dtype=tf.int32,
                    name="token_type_ids",
                ),
            },
            "text_b": {
                "input_ids": keras.Input(
                    shape=(sequence_len,), dtype=tf.int32, name="input_ids",
                ),
                "attention_mask": keras.Input(
                    shape=(sequence_len,),
                    dtype=tf.int32,
                    name="attention_mask",
                ),
                "token_type_ids": keras.Input(
                    shape=(sequence_len,),
                    dtype=tf.int32,
                    name="token_type_ids",
                ),
            },
        }


class StsbModel(tf.keras.Model):
    def __init__(
        self,
        model_name_path: str,
        sequence_len: int,
        pretrained_model_name: str,
        use_avg_pooled_model: bool = False,
        use_dropout: bool = True,
    ):
        """
        ALBERT STS-B model.

        Args:
            model_name_path (str): model_name_path
            sequence_len (int): sequence_len
            pretrained_model_name (str): pretrained_model_name
            use_avg_pooled_model (bool): use_avg_pooled_model
            use_dropout (bool): use_dropout
        """
        self.model_name_path = model_name_path
        self.sequence_len = sequence_len
        self.pretrained_model_name = pretrained_model_name
        self.use_dropout = use_dropout
        self.use_avg_pooled_model = use_avg_pooled_model
        super().__init__()
        self.pretrained_model = (
            PretrainedModelAvgPooling(
                self.model_name_path, use_dropout, self.pretrained_model_name
            )
            if self.use_avg_pooled_model
            else PretrainedModelCls(
                self.model_name_path, use_dropout, self.pretrained_model_name
            )
        )
        super().__init__()

        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        bias_initializer = tf.keras.initializers.zeros()
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
        output = self.pretrained_model(inputs)
        output = self.dense(output)
        output = tf.squeeze(output, [-1], name="output")
        return output

    def get_config(self) -> Dict[str, str]:
        """Update config."""
        config = super.get_config()
        config.update(
            {
                "model_name_path": self.model_name_path,
                "sequence_len": self.sequence_len,
                "pretrained_model_name": self.pretrained_model_name,
                "use_avg_pooled_model": self.use_avg_pooled_model,
                "use_dropout": self.use_dropout,
            }
        )
        return config

    @classmethod
    def sample_input(cls, sequence_len: int) -> Dict[str, tf.Tensor]:
        """
        Returns sample inputs.

        Args:
            sequence_len (int): sequence_len

        Returns:
            Dict[str, tf.Tensor]:
        """
        inputs = {
            "input_ids": keras.Input(
                shape=(sequence_len,), dtype=tf.int32, name="input_ids",
            ),
            "attention_mask": keras.Input(
                shape=(sequence_len,), dtype=tf.int32, name="attention_mask",
            ),
            "token_type_ids": keras.Input(
                shape=(sequence_len,), dtype=tf.int32, name="token_type_ids",
            ),
        }
        return inputs
