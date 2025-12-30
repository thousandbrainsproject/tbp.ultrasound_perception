# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running an experiment."""

from tbp.monty.frameworks.run_env import setup_env

setup_env()

# # Load all experiment configurations from local project
from tbp.monty.frameworks.run import main  # noqa: E402

from configs import CONFIGS  # noqa: E402


import copy
import logging
import os
import pprint
import time

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict

logger = logging.getLogger(__name__)


def merge_args(config, cmd_args=None):
    """Override experiment "config" parameters with command line args.

    Returns:
        Updated config with command line args.
    """
    if not cmd_args:
        return config

    exp_config = copy.deepcopy(config)
    exp_config.update(cmd_args.__dict__)
    del exp_config["experiments"]
    return exp_config


def print_config(config):
    """Print config with nice formatting if config_args.print_config is True."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(pprint.pformat(config))
    print("-" * 100)


def run(config):
    with config["experiment_class"](config) as exp:
        # TODO: Later will want to evaluate every x episodes or epochs
        # this could probably be solved with just setting the logging freqency
        # Since each trainng loop already does everything that eval does.
        if exp.do_train:
            print("---------training---------")
            exp.train()

        if exp.do_eval:
            print("---------evaluating---------")
            exp.evaluate()


def main(all_configs, experiments=None):
    """Use this as "main" function when running monty experiments.

    A typical project `run.py` should look like this::

        # Load all experiment configurations from local project
        from experiments import CONFIGS
        from tbp.monty.frameworks.run import main

        if __name__ == "__main__":
            main(all_configs=CONFIGS)

    Args:
        all_configs: Dict containing all available experiment configurations.
            Usually each project would have its own list of experiment
            configurations
        experiments: Optional list of experiments to run, used to bypass the
            command line args
    """
    cmd_args = None
    if not experiments:
        cmd_parser = create_cmd_parser(experiments=list(all_configs.keys()))
        cmd_args = cmd_parser.parse_args()
        experiments = cmd_args.experiments

        if cmd_args.quiet_habitat_logs:
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"

    for experiment in experiments:
        exp = all_configs[experiment]
        exp_config = merge_args(exp, cmd_args)  # TODO: is this really even necessary?
        exp_config = config_to_dict(exp_config)

        # Update run_name and output dir with experiment name
        # NOTE: wandb args are further processed in monty_experiment
        if not exp_config["logging_config"]["run_name"]:
            exp_config["logging_config"]["run_name"] = experiment

        if "probe_triggered" in experiment:
            # Prompt user for object name to create subdirectory
            object_name = input(
                "Enter the name of the object for this experiment: "
            ).strip()
            if not object_name:
                object_name = "unknown_object"

            # Sanitize object name for use in directory path
            object_name = "".join(
                c for c in object_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            object_name = object_name.replace(" ", "_")

            exp_config["logging_config"]["output_dir"] = os.path.join(
                exp_config["logging_config"]["output_dir"],
                exp_config["logging_config"]["run_name"],
                object_name,
            )
            # Update the save_path for the dataset to use output_dir
            exp_config["dataset_args"]["env_init_args"]["save_path"] = os.path.join(
                exp_config["logging_config"]["output_dir"], "observations"
            )

        else:
            exp_config["logging_config"]["output_dir"] = os.path.join(
                exp_config["logging_config"]["output_dir"],
                exp_config["logging_config"]["run_name"],
            )

        # If we are not running in parallel, this should always be False
        exp_config["logging_config"]["log_parallel_wandb"] = False
        print_config(exp_config)

        # Print config without running experiment
        if cmd_args is not None:
            if cmd_args.print_config:
                continue

        os.makedirs(exp_config["logging_config"]["output_dir"], exist_ok=True)
        start_time = time.time()
        run(exp_config)
        logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")


if __name__ == "__main__":
    main(all_configs=CONFIGS)
