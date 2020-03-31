import os
import time
import argparse
import yaml
from albert import classifier_utils
from albert import fine_tuning_utils
from albert import modeling
from preprocessing import SemEval
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu


def fine_tune_albert(config):
    tf.logging.set_verbosity(tf.logging.INFO)

    if (
        not config.get("do_train", False)
        and not config.get("do_eval", False)
        and not config.get("do_predict", False)
    ):
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True."
        )

    if not config.get("albert_config_file", None) and not config.get(
        "albert_hub_module_handle", None
    ):
        raise ValueError(
            "At least one of `--albert_config_file` and "
            "`--albert_hub_module_handle` must be set"
        )

    if config.get("albert_config_file", None):
        albert_config = modeling.AlbertConfig.from_json_file(
            config.get("albert_config_file", None)
        )
        if (
            config.get("max_seq_length", 512)
            > albert_config.max_position_embeddings
        ):
            raise ValueError(
                "Cannot use sequence length %d because the ALBERT model "
                "was only trained up to sequence length %d"
                % (
                    config.get("max_seq_length", 512),
                    albert_config.max_position_embeddings,
                )
            )
    else:
        albert_config = None  # Get the config from TF-Hub.

    tf.gfile.MakeDirs(config.get("output_dir", None))

    task_name = config.get("task_name", None).lower()

    processor = SemEval(config)
    label_list = processor.get_labels()

    tokenizer = fine_tuning_utils.create_vocab(
        vocab_file=config.get("vocab_file", None),
        do_lower_case=config.get("do_lower_case", True),
        spm_model_file=config.get("spm_model_file", None),
        hub_module=config.get("albert_hub_module_handle", None),
    )

    tpu_cluster_resolver = None
    if config.get("use_tpu", False) and config.get("tpu_name", None):
        tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
            config.get("tpu_name", None),
            zone=config.get("tpu_zone", None),
            project=config.get("gcp_project", None),
        )

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    if config.get("do_train", False):
        iterations_per_loop = int(
            min(
                config.get("iterations_per_loop", 1000),
                config.get("save_checkpoints_steps", 1000),
            )
        )
    else:
        iterations_per_loop = config.get("iterations_per_loop", 1000)
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=config.get("master", None),
        model_dir=config.get("output_dir", None),
        save_checkpoints_steps=int(
            config.get("save_checkpoints_steps", 1000)
        ),
        keep_checkpoint_max=0,
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=config.get("num_tpu_cores", 8),
            per_host_input_for_training=is_per_host,
        ),
    )

    train_examples = None
    if config.get("do_train", False):
        train_examples = processor.get_train_examples()
    model_fn = classifier_utils.model_fn_builder(
        albert_config=albert_config,
        num_labels=len(label_list),
        init_checkpoint=config.get("init_checkpoint", None),
        learning_rate=config.get("learning_rate", 5e-5),
        num_train_steps=config.get("train_step", 1000),
        num_warmup_steps=config.get("warmup_step", 0),
        use_tpu=config.get("use_tpu", False),
        use_one_hot_embeddings=config.get("use_tpu", False),
        task_name=task_name,
        hub_module=config.get("albert_hub_module_handle", None),
        optimizer=config.get("optimizer", None),
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = contrib_tpu.TPUEstimator(
        use_tpu=config.get("use_tpu", False),
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.get("train_batch_size", 32),
        eval_batch_size=config.get("eval_batch_size", 8),
        predict_batch_size=config.get("predict_batch_size", 8),
    )

    if config.get("do_train", False):
        cached_dir = config.get("cached_dir", None)
        if not cached_dir:
            cached_dir = config.get("output_dir", None)
        train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
        if not tf.gfile.Exists(train_file):
            classifier_utils.file_based_convert_examples_to_features(
                train_examples,
                label_list,
                config.get("max_seq_length", 512),
                tokenizer,
                train_file,
                task_name,
            )
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info(
            "  Batch size = %d", config.get("train_batch_size", 32)
        )
        tf.logging.info("  Num steps = %d", config.get("train_step", 1000))
        train_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=config.get("max_seq_length", 512),
            is_training=True,
            drop_remainder=True,
            task_name=task_name,
            use_tpu=config.get("use_tpu", False),
            bsz=config.get("train_batch_size", 32),
        )
        estimator.train(
            input_fn=train_input_fn, max_steps=config.get("train_step", 1000)
        )

    if config.get("do_eval", False):
        eval_examples = processor.get_dev_examples()
        num_actual_eval_examples = len(eval_examples)
        if config.get("use_tpu", False):
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % config.get("eval_batch_size", 8) != 0:
                eval_examples.append(classifier_utils.PaddingInputExample())

        cached_dir = config.get("cached_dir", None)
        if not cached_dir:
            cached_dir = config.get("output_dir", None)
        eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
        if not tf.gfile.Exists(eval_file):
            classifier_utils.file_based_convert_examples_to_features(
                eval_examples,
                label_list,
                config.get("max_seq_length", 512),
                tokenizer,
                eval_file,
                task_name,
            )

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info(
            "  Num examples = %d (%d actual, %d padding)",
            len(eval_examples),
            num_actual_eval_examples,
            len(eval_examples) - num_actual_eval_examples,
        )
        tf.logging.info(
            "  Batch size = %d", config.get("eval_batch_size", 8)
        )

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if config.get("use_tpu", False):
            assert len(eval_examples) % config.get("eval_batch_size", 8) == 0
            eval_steps = int(
                len(eval_examples) // config.get("eval_batch_size", 8)
            )

        eval_drop_remainder = True if config.get("use_tpu", False) else False
        eval_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=config.get("max_seq_length", 512),
            is_training=False,
            drop_remainder=eval_drop_remainder,
            task_name=task_name,
            use_tpu=config.get("use_tpu", False),
            bsz=config.get("eval_batch_size", 8),
        )

        best_trial_info_file = os.path.join(
            config.get("output_dir", None), "best_trial.txt"
        )

        def _best_trial_info():
            """Returns information about which checkpoints have been evaled so far."""
            if tf.gfile.Exists(best_trial_info_file):
                with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
                    (
                        global_step,
                        best_metric_global_step,
                        metric_value,
                    ) = best_info.read().split(":")
                    global_step = int(global_step)
                    best_metric_global_step = int(best_metric_global_step)
                    metric_value = float(metric_value)
            else:
                metric_value = -1
                best_metric_global_step = -1
                global_step = -1
            tf.logging.info(
                "Best trial info: Step: %s, Best Value Step: %s, "
                "Best Value: %s",
                global_step,
                best_metric_global_step,
                metric_value,
            )
            return global_step, best_metric_global_step, metric_value

        def _remove_checkpoint(checkpoint_path):
            for ext in ["meta", "data-00000-of-00001", "index"]:
                src_ckpt = checkpoint_path + ".{}".format(ext)
                tf.logging.info("removing {}".format(src_ckpt))
                tf.gfile.Remove(src_ckpt)

        def _find_valid_cands(curr_step):
            filenames = tf.gfile.ListDirectory(
                config.get("output_dir", None)
            )
            candidates = []
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    idx = ckpt_name.split("-")[-1]
                    if int(idx) > curr_step:
                        candidates.append(filename)
            return candidates

        output_eval_file = os.path.join(
            config.get("output_dir", None), "eval_results.txt"
        )

        if task_name == "sts-b":
            key_name = "pearson"
        elif task_name == "cola":
            key_name = "matthew_corr"
        else:
            key_name = "eval_accuracy"

        global_step, best_perf_global_step, best_perf = _best_trial_info()
        writer = tf.gfile.GFile(output_eval_file, "w")
        while global_step < config.get("train_step", 1000):
            steps_and_files = {}
            filenames = tf.gfile.ListDirectory(
                config.get("output_dir", None)
            )
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    cur_filename = os.path.join(
                        config.get("output_dir", None), ckpt_name
                    )
                    if cur_filename.split("-")[-1] == "best":
                        continue
                    gstep = int(cur_filename.split("-")[-1])
                    if gstep not in steps_and_files:
                        tf.logging.info(
                            "Add {} to eval list.".format(cur_filename)
                        )
                        steps_and_files[gstep] = cur_filename
            tf.logging.info("found {} files.".format(len(steps_and_files)))
            if not steps_and_files:
                tf.logging.info(
                    "found 0 file, global step: {}. Sleeping.".format(
                        global_step
                    )
                )
                time.sleep(60)
            else:
                for checkpoint in sorted(steps_and_files.items()):
                    step, checkpoint_path = checkpoint
                    if global_step >= step:
                        if (
                            best_perf_global_step != step
                            and len(_find_valid_cands(step)) > 1
                        ):
                            _remove_checkpoint(checkpoint_path)
                        continue
                    result = estimator.evaluate(
                        input_fn=eval_input_fn,
                        steps=eval_steps,
                        checkpoint_path=checkpoint_path,
                    )
                    global_step = result["global_step"]
                    tf.logging.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        tf.logging.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("best = {}\n".format(best_perf))
                    if result[key_name] > best_perf:
                        best_perf = result[key_name]
                        best_perf_global_step = global_step
                    elif len(_find_valid_cands(global_step)) > 1:
                        _remove_checkpoint(checkpoint_path)
                    writer.write("=" * 50 + "\n")
                    writer.flush()
                    with tf.gfile.GFile(
                        best_trial_info_file, "w"
                    ) as best_info:
                        best_info.write(
                            "{}:{}:{}".format(
                                global_step, best_perf_global_step, best_perf
                            )
                        )
        writer.close()

        for ext in ["meta", "data-00000-of-00001", "index"]:
            src_ckpt = "model.ckpt-{}.{}".format(best_perf_global_step, ext)
            tgt_ckpt = "model.ckpt-best.{}".format(ext)
            tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
            tf.io.gfile.rename(
                os.path.join(config.get("output_dir", None), src_ckpt),
                os.path.join(config.get("output_dir", None), tgt_ckpt),
                overwrite=True,
            )

    if config.get("do_predict", False):
        predict_examples = processor.get_test_examples()
        num_actual_predict_examples = len(predict_examples)
        if config.get("use_tpu", False):
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while (
                len(predict_examples) % config.get("predict_batch_size", 8)
                != 0
            ):
                predict_examples.append(classifier_utils.PaddingInputExample())

        predict_file = os.path.join(
            config.get("output_dir", None), "predict.tf_record"
        )
        classifier_utils.file_based_convert_examples_to_features(
            predict_examples,
            label_list,
            config.get("max_seq_length", 512),
            tokenizer,
            predict_file,
            task_name,
        )

        tf.logging.info("***** Running prediction*****")
        tf.logging.info(
            "  Num examples = %d (%d actual, %d padding)",
            len(predict_examples),
            num_actual_predict_examples,
            len(predict_examples) - num_actual_predict_examples,
        )
        tf.logging.info(
            "  Batch size = %d", config.get("predict_batch_size", 8)
        )

        predict_drop_remainder = (
            True if config.get("use_tpu", False) else False
        )
        predict_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=config.get("max_seq_length", 512),
            is_training=False,
            drop_remainder=predict_drop_remainder,
            task_name=task_name,
            use_tpu=config.get("use_tpu", False),
            bsz=config.get("predict_batch_size", 8),
        )

        checkpoint_path = os.path.join(
            config.get("output_dir", None), "model.ckpt-best"
        )
        result = estimator.predict(
            input_fn=predict_input_fn, checkpoint_path=checkpoint_path
        )

        output_predict_file = os.path.join(
            config.get("output_dir", None), "test_results.tsv"
        )
        output_submit_file = os.path.join(
            config.get("output_dir", None), "submit_results.tsv"
        )
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
        with tf.gfile.GFile(
            output_predict_file, "w"
        ) as pred_writer, tf.gfile.GFile(
            output_submit_file, "w"
        ) as sub_writer:
            sub_writer.write("index" + "\t" + "prediction\n")
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, (example, prediction)) in enumerate(
                zip(predict_examples, result)
            ):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = (
                    "\t".join(
                        str(class_probability)
                        for class_probability in probabilities
                    )
                    + "\n"
                )
                pred_writer.write(output_line)

                if task_name != "sts-b":
                    actual_label = label_list[int(prediction["predictions"])]
                else:
                    actual_label = str(prediction["predictions"])
                sub_writer.write(example.guid + "\t" + actual_label + "\n")
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


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
    print(config)
    fine_tune_albert(config)
