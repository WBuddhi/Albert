import os
from dataprocessor import DataProcessor, StsbProcessor
from typing import Tuple
from transformers import AutoTokenizer
from utils import import_fn


def generate_example_datasets(config: dict) -> Tuple:
    """
    Generate training, eval and test datasets.

    Args:
        config (dict): config

    Returns:
        Tuple: (train_dataset, eval_dataset, test_dataset, config)
    """
    input_seperated = config.get("inputs_seperated", False)
    use_spm = config.get("spm_model_file", False)
    do_lower_case = config.get("do_lower_case", False)
    normalize_scores = config.get("normalize_scores", True)
    seq_len = config.get("sequence_len", 512)
    module_name = config.get("preprocessor", None)

    processor = StsbProcessor(use_spm, do_lower_case, normalize_scores)
    function_name = "file_based_input_fn_builder"
    file_based_input_fn_builder = import_fn(module_name, function_name)

    (model_files, config) = create_train_eval_input_files(config, processor)

    model_datasets = {}
    for prefix, model_file in model_files.items():
        is_training = False
        if prefix == "train":
            is_training = True
        model_datasets[prefix] = file_based_input_fn_builder(
            model_file,
            seq_len,
            is_training=is_training,
            bsz=config.get(f"{prefix}_batch_size", 32),
        )
    return model_datasets, config


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
    normalize = config.get("normalize_scores", True)
    module_name = config.get("preprocessor", "None")
    seq_len = config.get("sequence_len", 512)
    function_name = "file_based_convert_examples_to_features"

    file_based_convert_examples_to_features = import_fn(
        module_name, function_name
    )
    if not cached_dir:
        cached_dir = config.get("output_dir", None)
    model_files = _generate_model_files(cached_dir, task_name)
    model_examples, config = _generate_model_examples(
        processor, data_dir, config
    )
    tokenizer = _get_tokenizer(config)
    for data_file, examples in zip(list(model_files.values()), model_examples):
        file_based_convert_examples_to_features(
            examples, seq_len, tokenizer, data_file, task_name,
        )
    return model_files, config


def _generate_model_files(cached_dir: str, task_name: str):

    model_files_suffix = ["_train", "_eval", "_test"]
    extension = ".tf_record"
    model_files = []
    model_files = {}
    for suffix in model_files_suffix:
        model_file = os.path.join(cached_dir, task_name + suffix + extension)
        model_files[suffix.strip("_")] = model_file
    return model_files


def _generate_model_examples(
    processor: DataProcessor, data_dir: str, config: dict
):
    train_examples = processor.get_train_examples(data_dir)
    eval_examples = processor.get_eval_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)
    config["train_size"] = len(train_examples)
    config["eval_size"] = len(eval_examples)
    config["test_size"] = len(test_examples)
    return [train_examples, eval_examples, test_examples], config


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
