import os
import csv
import tensorflow.compat.v1 as tf
from typing import List
import tokenization
from preprocess import InputExample


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

    def get_test(self, data_dir: str):
        """
        Gets a collection of `InputExample`s for the test set.

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
