import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from albert.classifier_utils import DataProcessor, InputExample


class SemEval(DataProcessor):
    def __init__(self, config):
        DataProcessor.__init__(
            self,
            use_spm=True if config.get("spm_model_file", None) else False,
            do_lower_case=config.get("do_lower_case", True),
        )
        data_dir = config.get("data_dir", None)
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")
        train_df = self._load_df(train_path)
        split = config.get("split", None)
        if split:
            self.train_df, self.val_df = train_test_split(
                train_df, test_size=split
            )
        else:
            self.train_df = train_df
            self.val_df = None
        self.test_df = self._load_df(test_path)

    def get_train_examples(self):
        return self._create_examples(self.train_df)

    def get_dev_examples(self):
        return self._create_examples(self.val_df)

    def get_test_examples(self):
        return self._create_examples(self.test_df)
    
    def get_labels(self):
        return ["0","1"]

    def _create_examples(self, df):
        examples = []
        guids = df.index
        df["score"] = df["score"].astype(float)
        df["sent1"] = df["sent1"].apply(self.process_text)
        df["sent2"] = df["sent2"].apply(self.process_text)
        sent1 = df["sent1"].tolist()
        sent2 = df["sent2"].tolist()
        score = df["score"].tolist()
        for guid, text_a, text_b, label in zip(guids, sent1, sent2, score):
            examples.append(
                InputExample(
                    guid=str(guid),
                    text_a=text_a,
                    text_b=text_b,
                    label=str(int(label)),
                )
            )
        return examples

    def _load_df(self, path):
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)
        df["score"] = df["score"].astype(float)
        return df
