import os
from dataprocessor import DataProcessor, StsbProcessor
from typing import Tuple
from transformers import AutoTokenizer

# from preprocessing.double_sent_preprocess import (
#    file_based_input_fn_builder,
#    file_based_convert_examples_to_features,
# )
from preprocessing.individual_sent_preprocess import (
    file_based_input_fn_builder,
    file_based_convert_examples_to_features,
)


def generate_example_datasets(config: dict) -> Tuple:
    """
    Generate training, eval and test datasets.

    Args:
        config (dict): config
    Returns:
        Tuple: (train_dataset, eval_dataset, test_dataset, config)
    """
    processor = StsbProcessor(
        config.get("spm_model_file", False),
        config.get("do_lower_case", False),
        config.get("normalize_labels", True),
    )
    seq_len = config.get("sequence_len", 512)
    (
        train_file,
        eval_file,
        test_file,
        config,
    ) = create_train_eval_input_files(config, processor)

    train_dataset = file_based_input_fn_builder(
        train_file,
        seq_len,
        is_training=True,
        bsz=config.get("train_batch_size", 32),
    )
    eval_dataset = file_based_input_fn_builder(
        eval_file,
        seq_len,
        is_training=False,
        bsz=config.get("eval_batch_size", 32),
    )
    test_dataset = file_based_input_fn_builder(
        test_file,
        seq_len,
        is_training=False,
        bsz=config.get("test_batch_size", 32),
    )
    return train_dataset, eval_dataset, test_dataset, config


def create_train_eval_input_files(
    config: dict, processor: DataProcessor
) -> Tuple[str]:
    """
    Create training and eval input files.

    Args:
        config (dict): config
        processor (DataProcessor): processor

    Returns:
        Tuple[str]: (training data file,
            evaluation data file,
            test data file)
    """
    cached_dir = config.get("cached_dir", None)
    task_name = config.get("task_name", "Experiment")
    data_dir = config.get("data_dir", "")
    model_step_names = ["train", "eval", "test"]
    if not cached_dir:
        cached_dir = config.get("output_dir", None)
    train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
    train_examples = processor.get_train_examples(data_dir)
    config["train_size"] = len(train_examples)
    eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
    eval_examples = processor.get_eval_examples(data_dir)
    config["eval_size"] = len(eval_examples)
    test_file = os.path.join(cached_dir, task_name + "_test.tf_record")
    test_examples = processor.get_test_examples(data_dir)
    config["test_size"] = len(test_examples)
    tokenizer = _get_tokenizer(config)
    for data_file, examples in zip(
        (train_file, eval_file, test_file),
        (train_examples, eval_examples, test_examples),
    ):
        file_based_convert_examples_to_features(
            examples,
            config.get("sequence_len", 512),
            tokenizer,
            data_file,
            task_name,
        )
    return train_file, eval_file, test_file, config


def _get_tokenizer(config: dict) -> AutoTokenizer:
    """
    Get tokenizer.

    Args:
        config (dict): config

    Returns:
        FullTokenizer:
    """
    return AutoTokenizer.from_pretrained(
        config.get("transformer_name_path", None)
    )
