import os
import time
import argparse
import yaml
from albert.run_classifier import FLAGS, main, flags
import tensorflow.compat.v1 as tf
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="config file path",
        default="./config.yaml",
        type=str,
    )
    args = parser.parse_args()
    config = {}
    with open(args.config_file, "r") as stream:
        config = {**config, **yaml.safe_load(stream)}

    input_file = os.path.join(config["data_dir"], "STS-B", "train.tsv")
    quotechar = None
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    train_examples = len(lines)
    config["train_step"] = int(
        train_examples
        / config["train_batch_size"]
        * config["num_train_epochs"]
    )
    config["warmup_step"] = int(
        config["warmup_proportion"] * config["train_step"]
    )
    print(config)

    for key, value in config.items():
        if key in ["num_train_epochs", "warmup_proportion"]:
            continue
        setattr(FLAGS, key, value)

    tf.app.run()
