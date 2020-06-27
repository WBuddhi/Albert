"""
Utility functions.
"""
import yaml
from importlib import import_module


def read_yaml_config(config_file: str) -> dict:
    """
    Read yaml config file.

    Args:
        config_file (str): config_file

    Returns:
        dict: config file
    """
    config = {}
    with open(config_file, "r") as stream:
        config = {**config, **yaml.safe_load(stream)}
    return config


def import_fn(module_name: str, function: str) -> object:
    """
    Dynamic module importer.

    Args:
        module_name (str): module_name
        function (str): function

    Returns:
        object:
    """
    return getattr(import_module(module_name), function)
