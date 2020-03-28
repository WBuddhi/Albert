import pandas as pd
from pathlib import Path
from sklearn.model_selection import  train_test_split
from ALBERT.classifier_utils import DataProcessor, InputExample

class SemEval(DataProcessor):
    def __init__(self, train_path, test_path, split=0.25):
        train_df = _load_df(train_path)
        DataProcessor.__init__(self, use_spm = True, do_lower_case = False)
        if split:
            self.train_df, self.val_df = train_test_split(train_df, split=0.25)
        else:
            self.train_df = train_df
            self.val_df = None
        self.test_df = _load_df(test_path)

    def get_train_examples(self):
        return self._create_examples(self.train_df)

    def get_val_examples(self):
        return self._create_examples(self.val_df)

    def get_test_examples(self):
        return self._create_examples(self.test)

    def _create_examples(self, df):
        examples = []
        guids = df.index
        sent1 = df["sent1"].to_list()
        sent2 = df["sent2"].to_list()
        score = df["score"].to_list()
        for guid, text_a, text_b, label in zip(guids, sent1, sent2, score):
            examples.append(InputExample(guid = guid, text_a = text_a, text_b = text_b, label=str(int(label))))
        return examples

    def _load_df(self, path):
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)
        df['score'] = df['score'].astype(float)
        return df

