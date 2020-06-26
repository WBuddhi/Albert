"""
Utility functions.
"""
import yaml
from importlib import import_module


def read_yaml_config(config_file):
    config = {}
    with open(config_file, "r") as stream:
        config = {**config, **yaml.safe_load(stream)}
    return config

def import_fn(module_name: str, function: str):
    return getattr(import_module(module_name), function)
