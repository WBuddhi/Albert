"""
Methods in this file pre-processing the input to generate an output
of the following format:
    input_ids_a: [sent_a]
    attention_mask_a: [sent_a]
    token_type_ids_a: [sent_a]
    input_ids_b: [sent_b]
    attention_masks_b: [sent_b]
    token_type_ids_b: [sent_b]

This is more suited when Albert is used in a Siemese configuration.
"""
import tensorflow.compat.v1 as tf
import collections
from typing import Any, List, Callable, Tuple
from preprocessing.utils import *


def convert_single_example(
    ex_index: int,
    example: InputExample,
    max_seq_length: int,
    tokenizer: object,
    task_name: str,
) -> InputFeatures:
    """
    Converts a single `InputExample` to separate `InputFeatures`

    Args:
        ex_index (int): ex_index
        example (InputExample): example
        max_seq_length (int): max_seq_length
        tokenizer (object): tokenizer
        task_name (str): task_name

    Returns:
        InputFeatures:
    """
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    tokens_a, tokens_b = _truncate_seq_pair(
        tokens_a, tokens_b, max_seq_length - 2
    )
    input_ids_a, attention_mask_a, token_type_ids_a = create_albert_input(
        tokens_a=tokens_a, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    input_ids_b, attention_mask_b, token_type_ids_b = create_albert_input(
        tokens_a=tokens_b, tokenizer=tokenizer, max_seq_length=max_seq_length
    )

    label_id = example.label

    if ex_index < 3:
        tf.logging.debug("*** Example ***")
        tf.logging.debug("**Sentence a**")
        tf.logging.debug("guid: %s" % (example.guid))
        tf.logging.debug("tokens: %s" % " ".join(tokens_a))
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids_a])
        )
        tf.logging.debug(
            "attention_mask: %s" % " ".join([str(x) for x in attention_mask_a])
        )
        tf.logging.debug(
            "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids_a])
        )
        tf.logging.debug("**Sentence b**")
        tf.logging.debug("tokens: %s" % " ".join(tokens_b))
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids_b])
        )
        tf.logging.debug(
            "attention_mask: %s" % " ".join([str(x) for x in attention_mask_b])
        )
        tf.logging.debug(
            "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids_b])
        )
        tf.logging.debug("label: %s (id = %f)" % (example.label, label_id))

    feature = InputSepFeatures(
        input_ids_a=input_ids_a,
        attention_mask_a=attention_mask_a,
        token_type_ids_a=token_type_ids_a,
        input_ids_b=input_ids_b,
        attention_mask_b=attention_mask_b,
        token_type_ids_b=token_type_ids_b,
        label_id=label_id,
        is_real_example=True,
    )
    return feature


def file_based_input_fn_builder(
    input_file: str,
    seq_length: int,
    is_training: bool,
    drop_remainder: str = True,
    multiple: int = 1,
    bsz: int = 32,
) -> Callable:
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.

    Args:
        input_file (str): input_file
        seq_length (int): seq_length
        is_training (bool): is_training
        drop_remainder (str): drop_remainder
        multiple (int): multiple
        bsz (int): bsz

    Returns:
        Callable:
    """
    labeltype = tf.float32

    name_to_features = {
        "input_ids_a": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "attention_mask_a": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "token_type_ids_a": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "input_ids_b": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "attention_mask_b": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "token_type_ids_b": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "label_id": tf.FixedLenFeature([], labeltype),
    }

    def _decode_record(
        record: tf.data.TFRecordDataset, name_to_features: dict
    ) -> object:
        """
        Decodes a record to a TensorFlow example.

        tf.Example only supports tf.int64, but the TPU only
          supports tf.int32.
        So cast all int64 to int32.

        Args:
            record (tf.data.TFRecordDataset): record
            name_to_features (dict): name_to_features

        Returns:
            tf.Example:
        """
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32, name=name)
            example[name] = t

        inputs = {
            "text_a": {
                "input_ids": example["input_ids_a"],
                "attention_mask": example["attention_mask_a"],
                "token_type_ids": example["token_type_ids_a"],
            },
            "text_b": {
                "input_ids": example["input_ids_b"],
                "attention_mask": example["attention_mask_b"],
                "token_type_ids": example["token_type_ids_b"],
            },
        }

        return (inputs, example["label_id"])

    def input_fn():
        """
        The actual input function.

        For training, we want a lot of parallel reading and shuffling.
        For eval, we want no shuffling and parallel reading doesn't matter.
        """

        batch_size = bsz

        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(
                buffer_size=100, reshuffle_each_iteration=True
            )
        dataset = dataset.map(
            lambda record: _decode_record(record, name_to_features)
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        return dataset

    return input_fn()


def file_based_convert_examples_to_features(
    examples: InputExample,
    max_seq_length: int,
    tokenizer: object,
    output_file: str,
    task_name: str,
):
    """
    Convert a set of `InputExample`s to a TFRecord file.

    Args:
        examples (InputExample): examples
        max_seq_length (int): max_seq_length
        tokenizer (object): tokenizer
        output_file (str): output_file
        task_name (str): task_name
    """

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" % (ex_index, len(examples))
            )

        feature = convert_single_example(
            ex_index, example, max_seq_length, tokenizer, task_name
        )

        features = collections.OrderedDict()
        features["input_ids_a"] = create_int_feature(feature.input_ids_a)
        features["attention_mask_a"] = create_int_feature(
            feature.attention_mask_a
        )
        features["token_type_ids_a"] = create_int_feature(
            feature.token_type_ids_a
        )
        features["input_ids_b"] = create_int_feature(feature.input_ids_b)
        features["attention_mask_b"] = create_int_feature(
            feature.attention_mask_b
        )
        features["token_type_ids_b"] = create_int_feature(
            feature.token_type_ids_b
        )
        features["label_id"] = create_float_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)]
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


def _truncate_seq_pair(
    tokens_a: List[int], tokens_b: List[int], max_seq_length: int
):
    """
    Truncates a sequence pair in place to the maximum length.

    Args:
        tokens_a (List[int]): tokens_a
        tokens_b (List[int]): tokens_b
        max_seq_length (int): max_seq_length
    """
    for tokens in (tokens_a, tokens_b):
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0 : (max_seq_length - 2)]
    return tokens_a, tokens_b
