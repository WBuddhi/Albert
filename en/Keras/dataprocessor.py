import os
import csv
import six
import unicodedata
import tensorflow.compat.v1 as tf
from typing import List
from preprocessing.utils import InputExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(
        self, use_spm: bool, do_lower_case: bool, normalize: bool = False
    ):
        """
        Initializes DataProcessor.

        Args:
            use_spm (bool): use_spm
            do_lower_case (bool): do_lower_case
        """
        super(DataProcessor, self).__init__()
        self.normalize = normalize
        self.use_spm = use_spm
        self.do_lower_case = do_lower_case

    def get_train_examples(self, data_dir: str):
        """
        Gets a collection of Examples for the train set.

        Args:
            data_dir (str): data_dir
        """
        raise NotImplementedError()

    def get_eval_examples(self, data_dir: str):
        """
        Gets a collection of Examples for the test set.

        Args:
            data_dir (str): data_dir
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir: str):
        """
        Gets a collection of Examples for prediction.

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
            return self.preprocess_text(text, lower=self.do_lower_case)
        return self.convert_to_unicode(text)

    @classmethod
    def preprocess_text(
        cls, inputs: str, remove_space: bool = True, lower: bool = False
    ) -> str:
        """Preprocess data by removing extra space and normalize data.

        Args:
            inputs (str): inputs
            remove_space (bool): remove_space
            lower (bool): lower

        Returns:
            str:
        """
        outputs = inputs
        if remove_space:
            outputs = " ".join(inputs.strip().split())

        if six.PY2 and isinstance(outputs, str):
            try:
                outputs = six.ensure_text(outputs, "utf-8")
            except UnicodeDecodeError:
                outputs = six.ensure_text(outputs, "latin-1")

        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if lower:
            outputs = outputs.lower()

        return outputs

    @classmethod
    def convert_to_unicode(cls, text: str) -> str:
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input.

        Args:
            text (str): text

        Returns:
            str:
        """
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return six.ensure_text(text, "utf-8", "ignore")
            raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return six.ensure_text(text, "utf-8", "ignore")
            elif isinstance(text, six.text_type):
                return text
            raise ValueError("Unsupported string type: %s" % (type(text)))
        raise ValueError("Not running on Python2 or Python 3?")


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

    def get_eval_examples(
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
            "eval",
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
        """
        Get labels.

        Args:

        Returns:
            List[None]:
        """
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
                if self.normalize:
                    label = (float(line[-1]) - 0.0) / (5.0)
            else:
                label = 0
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label
                )
            )
        return examples
