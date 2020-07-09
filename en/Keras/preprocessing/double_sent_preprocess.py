"""
Methods in this file pre-processing the input to generate an output
of the following format:
    input_ids: [sent_a,sent_b]
    attention_masks: [sent_a, sent_b]
    token_type_ids: [sent_a, sent_b]

This is the default Albert input configuration, sent_b can be None.
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
    Converts a single `InputExample` into a single `InputFeatures`.

    Modifies `tokens_a` and `tokens_b` in place so that the total
    length is less than the specified length.
        Single sentence: Account for [CLS], [SEP], [SEP] with "- 3"
        Double sentence: Account for [CLS] and [SEP] with "- 2"

    Args:
        ex_index (int): ex_index
        example (InputExample): example
        max_seq_length (int): max_seq_length
        tokenizer (object): tokenizer
        task_name (str): task_name

    Returns:
        InputFeatures:
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0 for i in range(max_seq_length)],
            attention_mask=[0 for i in range(max_seq_length)],
            token_type_ids=[0 for i in range(max_seq_length)],
            label_id=0,
            is_real_example=False,
        )

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    elif len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    input_ids, attention_mask, token_type_ids = create_albert_input(
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    label_id = example.label

    tokens = tokens_a
    tokens.extend(tokens_b)
    if ex_index < 5:
        tf.logging.debug("*** Example ***")
        tf.logging.debug("guid: %s" % (example.guid))
        tf.logging.debug("tokens: %s" % " ".join(tokens))
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids])
        )
        tf.logging.debug(
            "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
        )
        tf.logging.debug(
            "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
        )
        tf.logging.debug("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
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
        "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "attention_mask": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "token_type_ids": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "label_ids": tf.FixedLenFeature([], labeltype),
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
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "token_type_ids": example["token_type_ids"],
        }

        return (inputs, example["label_ids"])

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
            dataset = dataset.shuffle(buffer_size=100)
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
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["attention_mask"] = create_int_feature(feature.attention_mask)
        features["token_type_ids"] = create_int_feature(feature.token_type_ids)
        features["label_ids"] = create_float_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)]
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


def _truncate_seq_pair(
    tokens_a: List[int], tokens_b: List[int], max_length: int
):
    """
    Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.

    Args:
        tokens_a (List[int]): tokens_a
        tokens_b (List[int]): tokens_b
        max_length (int): max_length
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
