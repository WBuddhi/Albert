import yaml


def read_yaml_config(config_file):
    config = {}
    with open(config_file, "r") as stream:
        config = {**config, **yaml.safe_load(stream)}
    return config
