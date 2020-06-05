import tensorflow.compat.v1 as tf
import collections

import tokenization
from typing import Any, List, Callable, Tuple


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(
        self, guid: int, text_a: str, text_b: str = None, label: Any = None
    ):
        """
        Constructs a InputExample.

        Args:
            guid (int): Unique id for the example.
            text_a (str): The untokenized text of the first sequence. For
                single sequence tasks, only this sequence must be specified.
            text_b (str): (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label (Any): (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.

        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids: List(int),
        input_mask: List(int),
        segment_ids: List(int),
        label_id: List(Any),
        guid: int = None,
        example_id: int = None,
        is_real_example: bool = True,
    ):
        """
        Create Input Feature

        Args:
            input_ids (List(int)): input_ids
            input_mask (List(int)): input_mask
            segment_ids (List(int)): segment_ids
            label_id (List(Any)): label_id
            guid (int): guid
            example_id (int): example_id
            is_real_example (bool): is_real_example
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_id = example_id
        self.guid = guid
        self.is_real_example = is_real_example


class InputSepFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids_a: List(int),
        input_mask_a: List(int),
        segment_ids_a: List(int),
        input_ids_b: List(int),
        input_mask_b: List(int),
        segment_ids_b: List(int),
        label_id: List(Any),
        guid: int = None,
        example_id: int = None,
        is_real_example: bool = True,
    ):
        """
        Create Input Feature for Sentences separately.

        Args:
            input_ids_a (List(int)): input_ids_a
            input_mask_a (List(int)): input_mask_a
            segment_ids_a (List(int)): segment_ids_a
            input_ids_b (List(int)): input_ids_b
            input_mask_b (List(int)): input_mask_b
            segment_ids_b (List(int)): segment_ids_b
            label_id (List(Any)): label_id
            guid (int): guid
            example_id (int): example_id
            is_real_example (bool): is_real_example
        """
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b
        self.label_id = label_id
        self.example_id = example_id
        self.guid = guid
        self.is_real_example = is_real_example


class PaddingInputExample(object):
    """
    Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.
    """


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
        "input_word_ids": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "label_ids": tf.FixedLenFeature([], labeltype),
    }

    def _decode_record(
        record: tf.data.TFRecordDataset, name_to_features: dict
    ) -> object:
        """
        Decodes a record to a TensorFlow example.

        Args:
            record (tf.data.TFRecordDataset): record
            name_to_features (dict): name_to_features

        Returns:
            tf.Example:
        """
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only
        #   supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32, name=name)
            example[name] = t

        return (example, example["label_ids"])

    def input_fn():
        """The actual input function."""

        batch_size = bsz

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
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


def _create_int_feature(values: List(int)) -> tf.train.Feature:
    """
    Create integer feature.

    Args:
        values (List(int)): values

    Returns:
        )tf.train.Feature:
    """
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values))
    )
    return feature


def _create_float_feature(values: List(float)) -> tf.train.Feature:
    """
    Create float feature.

    Args:
        values (List(float)): values

    Returns:
        )tf.train.Feature:
    """
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values))
    )
    return feature


def file_based_convert_examples_to_features(
    examples: InputExample,
    label_list: List,
    max_seq_length: int,
    tokenizer: tokenization.FullTokenizer,
    output_file: str,
    task_name: str,
):
    """
    Convert a set of `InputExample`s to a TFRecord file.

    Args:
        examples (InputExample): examples
        label_list (List): label_list
        max_seq_length (int): max_seq_length
        tokenizer (tokenization.FullTokenizer): tokenizer
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
            ex_index, example, label_list, max_seq_length, tokenizer, task_name
        )

        features = collections.OrderedDict()
        features["input_word_ids"] = _create_int_feature(feature.input_ids)
        features["input_mask"] = _create_int_feature(feature.input_mask)
        features["segment_ids"] = _create_int_feature(feature.segment_ids)
        features["label_ids"] = _create_float_feature([feature.label_id])
        features["is_real_example"] = _create_int_feature(
            [int(feature.is_real_example)]
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_sep_input_fn_builder(
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
        "input_word_ids_a": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "input_mask_a": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "segment_ids_a": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "label_ids_a": tf.FixedLenFeature([], labeltype),
        "input_word_ids_b": tf.FixedLenFeature(
            [seq_length * multiple], tf.int64
        ),
        "input_mask_b": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "segment_ids_b": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "label_ids_b": tf.FixedLenFeature([], labeltype),
    }

    def _decode_record(
        record: tf.data.TFRecordDataset, name_to_features: dict
    ) -> object:
        """
        Decodes a record to a TensorFlow example.

        Args:
            record (tf.data.TFRecordDataset): record
            name_to_features (dict): name_to_features

        Returns:
            tf.Example:
        """
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only
        #   supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32, name=name)
            example[name] = t

        inputs = {
            "text_a": {
                "input_word_ids": example["input_word_ids_a"],
                "input_mask": example["input_mask_a"],
                "segment_ids": example["segment_ids"],
            },
            "text_b": {
                "input_word_ids": example["input_word_ids_b"],
                "input_mask": example["input_mask_b"],
                "segment_ids": example["segment_ids_b"],
            },
        }

        return (example, example["label_ids"])

    def input_fn():
        """The actual input function."""

        batch_size = bsz

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
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


def file_based_convert_separated_examples_to_features(
    examples: InputExample,
    label_list: List,
    max_seq_length: int,
    tokenizer: tokenization.FullTokenizer,
    output_file: str,
    task_name: str,
):
    """
    Convert a set of `InputExample`s to a TFRecord file.

    Args:
        examples (InputExample): examples
        label_list (List): label_list
        max_seq_length (int): max_seq_length
        tokenizer (tokenization.FullTokenizer): tokenizer
        output_file (str): output_file
        task_name (str): task_name
    """

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" % (ex_index, len(examples))
            )

        feature = convert_example_seperately(
            ex_index, example, label_list, max_seq_length, tokenizer, task_name
        )

        features = collections.OrderedDict()
        features["input_word_ids_a"] = _create_int_feature(feature.input_ids_a)
        features["input_mask_a"] = _create_int_feature(feature.input_mask_a)
        features["segment_ids_a"] = _create_int_feature(feature.segment_ids_a)
        features["input_word_ids_b"] = _create_int_feature(feature.input_ids_b)
        features["input_mask_b"] = _create_int_feature(feature.input_mask_b)
        features["segment_ids_b"] = _create_int_feature(feature.segment_ids_b)
        features["label_ids"] = _create_float_feature([feature.label_id])
        features["is_real_example"] = _create_int_feature(
            [int(feature.is_real_example)]
        )

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        writer.write(tf_example.SerializeToString())
    writer.close()


def _truncate_seq_pair(
    tokens_a: List(int), tokens_b: List(int), max_length: int
):
    """
    Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal percent
    of tokens from each, since if one sequence is very short then each token
    that's truncated likely contains more information than a longer sequence.
    Args:
        tokens_a (List(int)): tokens_a
        tokens_b (List(int)): tokens_b
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


def _truncate_seq_pair_separetly(
    tokens_a: List(int), tokens_b: List(int), max_seq_length: int
):
    """
    Truncates a sequence pair in place to the maximum length.

    Args:
        tokens_a (List(int)): tokens_a
        tokens_b (List(int)): tokens_b
        max_seq_length (int): max_seq_length
    """
    for tokens in (tokens_a, tokens_b):
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0 : (max_seq_length - 2)]
    return tokens_a, tokens_b


def _create_albert_input(
    tokens_a: List,
    tokenizer: tokenization.FullTokenizer,
    max_seq_length: int,
    tokens_b: List = None,
) -> Tuple(List(int), List(int), List(int)):
    """
    Create Albert input.

    The convention in ALBERT is:
    (a) For sequence pairs:
     tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
     type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    (b) For single sequences:
     tokens:   [CLS] the dog is hairy . [SEP]
     type_ids: 0     0   0   0  0     0 0
    
    Where "type_ids" are used to indicate whether this is the first
    sequence or the second sequence. The embedding vectors for `type=0` and
    `type=1` were learned during pre-training and are added to the
    embedding vector (and position vector). This is not *strictly* necessary
    since the [SEP] token unambiguously separates the sequences, but it makes
    it easier for the model to learn the concept of sequences.
    
    For classification tasks, the first vector (corresponding to [CLS]) is
    used as the "sentence vector". Note that this only makes sense because
    the entire model is fine-tuned.

    Args:
        tokens_a (List): tokens_a
        tokens_b (List): tokens_b
        tokenizer (tokenization.FullTokenizer): tokenizer
        max_seq_length (int): max_seq_length
    Returns:
        Tuple(List(int),List(int),List(int)): input_ids, input_mask, segment_ids.
    """
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_example_seperately(
    ex_index: int,
    example: InputExample,
    max_seq_length: int,
    tokenizer: tokenization.FullTokenizer,
    task_name: str,
) -> InputFeatures:
    """
    Converts a single `InputExample` to separate `InputFeatures`

    Args:
        ex_index (int): ex_index
        example (InputExample): example
        max_seq_length (int): max_seq_length
        tokenizer (tokenization.FullTokenizer): tokenizer
        task_name (str): task_name

    Returns:
        InputFeatures:
    """
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    tokens_a, tokens_b = _truncate_seq_pair_separetly(
        tokens_a, tokens_b, max_seq_length - 2
    )
    input_ids_a, input_mask_a, segment_ids_a = _create_albert_input(
        tokens_a=tokens_a, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    input_ids_b, input_mask_b, segment_ids_b = _create_albert_input(
        tokens_a=tokens_b, tokenizer=tokenizer, max_seq_length=max_seq_length
    )

    label_id = example.label

    if ex_index < 3:
        tf.logging.debug("*** Example ***")
        tf.logging.debug("**Sentence a**")
        tf.logging.debug("guid: %s" % (example.guid))
        tf.logging.debug(
            "tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in tokens_a])
        )
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids_a])
        )
        tf.logging.debug(
            "input_mask: %s" % " ".join([str(x) for x in input_mask_a])
        )
        tf.logging.debug(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids_a])
        )
        tf.logging.debug("**Sentence b**")
        tf.logging.debug(
            "tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in tokens_b])
        )
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids_b])
        )
        tf.logging.debug(
            "input_mask: %s" % " ".join([str(x) for x in input_mask_b])
        )
        tf.logging.debug(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids_b])
        )
        tf.logging.debug("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids_a=input_ids_a,
        input_mask_a=input_mask_a,
        segment_ids_a=segment_ids_a,
        input_ids_b=input_ids_b,
        input_mask_b=input_mask_b,
        segment_ids_b=segment_ids_b,
        label_id=label_id,
        is_real_example=True,
    )
    return feature


def convert_single_example(
    ex_index: int,
    example: InputExample,
    max_seq_length: int,
    tokenizer: tokenization.FullTokenizer,
    task_name: str,
) -> InputFeatures:
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    
    Args:
        ex_index (int): ex_index
        example (InputExample): example
        max_seq_length (int): max_seq_length
        tokenizer (tokenization.FullTokenizer): tokenizer
        task_name (str): task_name

    Returns:
        InputFeatures:
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False,
        )

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0 : (max_seq_length - 2)]

    input_ids, input_mask, segment_ids = _create_albert_input(
        tokens_a, tokens_b, tokenizer, max_seq_length
    )
    label_id = example.label

    tokens = tokens_a.extend(tokens_b)
    if ex_index < 5:
        tf.logging.debug("*** Example ***")
        tf.logging.debug("guid: %s" % (example.guid))
        tf.logging.debug(
            "tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in tokens])
        )
        tf.logging.debug(
            "input_ids: %s" % " ".join([str(x) for x in input_ids])
        )
        tf.logging.debug(
            "input_mask: %s" % " ".join([str(x) for x in input_mask])
        )
        tf.logging.debug(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids])
        )
        tf.logging.debug("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True,
    )
    return feature
