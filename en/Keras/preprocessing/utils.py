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
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        label_id: List[Any],
        guid: int = None,
        example_id: int = None,
        is_real_example: bool = True,
    ):
        """
        Create Input Feature

        Args:
            input_ids (List[int]): input_ids
            input_mask (List[int]): input_mask
            segment_ids (List[int]): segment_ids
            label_id (List[Any]): label_id
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
        input_ids_a: List[int],
        input_mask_a: List[int],
        segment_ids_a: List[int],
        input_ids_b: List[int],
        input_mask_b: List[int],
        segment_ids_b: List[int],
        label_id: List[Any],
        guid: int = None,
        example_id: int = None,
        is_real_example: bool = True,
    ):
        """
        Create Input Feature for Sentences separately.

        Args:
            input_ids_a (List[int]): input_ids_a
            input_mask_a (List[int]): input_mask_a
            segment_ids_a (List[int]): segment_ids_a
            input_ids_b (List[int]): input_ids_b
            input_mask_b (List[int]): input_mask_b
            segment_ids_b (List[int]): segment_ids_b
            label_id (List[Any]): label_id
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


def create_albert_input(
    tokens_a: List,
    tokenizer: tokenization.FullTokenizer,
    max_seq_length: int,
    tokens_b: List = None,
) -> Tuple[List[int], List[int], List[int]]:
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
        Tuple[List[int],List[int],List[int]]: input_ids, input_mask, segment_ids.
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


def create_int_feature(values: List[int]) -> tf.train.Feature:
    """
    Create integer feature.

    Args:
        values (List[int]): values

    Returns:
        )tf.train.Feature:
    """
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values))
    )
    return feature


def create_float_feature(values: List[float]) -> tf.train.Feature:
    """
    Create float feature.

    Args:
        values (List[float]): values

    Returns:
        )tf.train.Feature:
    """
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values))
    )
    return feature
