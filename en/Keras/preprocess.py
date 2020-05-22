import os
import csv
import tensorflow.compat.v1 as tf

import tokenization
from typing import Any, List, Callable


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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, use_spm: bool, do_lower_case: bool):
        """
        Initializes DataProcessor.

        Args:
            use_spm (bool): use_spm
            do_lower_case (bool): do_lower_case
        """
        super(DataProcessor, self).__init__()
        self.use_spm = use_spm
        self.do_lower_case = do_lower_case

    def get_train_examples(self, data_dir: str):
        """
        Gets a collection of `InputExample`s for the train set.

        Args:
            data_dir (str): data_dir
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir: str):
        """
        Gets a collection of `InputExample`s for the dev set.

        Args:
            data_dir (str): data_dir
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir: str):
        """
        Gets a collection of `InputExample`s for prediction.

        Args:
            data_dir (str): data_dir
        """
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file: str, quotechar: str = None) -> List:
        """
        Reads a tab separated value file.

        Args:
            input_file (str): input_file
            quotechar (str): quotechar

        Returns:
            List:
        """
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def process_text(self, text: str) -> str:
        """
        Preprocess input test.

        Args:
            text (str): text

        Returns:
            str:
        """
        if self.use_spm:
            return tokenization.preprocess_text(text, lower=self.do_lower_case)
        else:
            return tokenization.convert_to_unicode(text)


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(
        self, data_dir: str, folder_name="STS-B"
    ) -> List[InputExample]:
        """
        Get training examples.

        Args:
            data_dir (str): data_dir
            folder_name:

        Returns:
            List[InputExample]:
        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, folder_name, "train.tsv")),
            "train",
        )

    def get_dev_examples(
        self, data_dir: str, folder_name: str = "STS-B"
    ) -> List[InputExample]:
        """
        Get evaluation examples.

        Args:
            data_dir (str): data_dir
            folder_name (str): folder_name

        Returns:
            List[InputExample]:
        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, folder_name, "dev.tsv")),
            "dev",
        )

    def get_test_examples(
        self, data_dir: str, folder_name: str = "STS-B"
    ) -> List[InputExample]:
        """
        Get test examples.

        Args:
            data_dir (str): data_dir
            folder_name (str): folder_name

        Returns:
            List[InputExample]:
        """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, folder_name, "test.tsv")),
            "test",
        )

    def get_labels(self) -> List[None]:
        return [None]

    def _create_examples(
        self, lines: List[str], set_type: str
    ) -> List[InputExample]:
        """
        Create training examples.

        Args:
            lines (List[str]): lines
            set_type (str): set_type

        Returns:
            List[InputExample]:
        """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = self.process_text(line[0])
            text_a = self.process_text(line[7])
            text_b = self.process_text(line[8])
            if set_type != "test":
                label = float(line[-1])
            else:
                label = 0
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label
                )
            )
        return examples


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
        "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
        "label_ids": tf.FixedLenFeature([], labeltype),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(
        record: tf.data.TFRecordDataset, name_to_features: dict
    ) -> tf.Example:
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
                t = tf.to_int32(t)
            example[name] = t

        return example

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

    return input_fn
