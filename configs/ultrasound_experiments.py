# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import math
import os
from copy import deepcopy
import sys

import numpy as np
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    ParallelEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import EvalExperimentArgs
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
)
from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataset
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_system import MotorSystem

from custom_classes.config import PlottingConfig
from custom_classes.dataloader import UltrasoundDataLoader
from custom_classes.environment import (
    JSONDatasetUltrasoundEnvironment,
    UltrasoundEnvironment,
)
from custom_classes.experiment import UltrasoundExperiment
from custom_classes.monty_class import MontyForEvidenceGraphMatchingWithGoalStateServer
from custom_classes.motor_policy import UltrasoundMotorPolicy
from custom_classes.probe_triggered_environment import (
    ProbeTriggeredUltrasoundEnvironment,
)
from custom_classes.sensor_module import UltrasoundSM

from .config_utils import import_config_from_monty

# Ultrasound experiments can use the models trained in simulation when inferring objects
# in the real world.
pretrained_dir = import_config_from_monty("defaults.py", "pretrained_dir")
model_path_tbp_robot_lab = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_tbp_robot_lab/pretrained/",
)

# ===== BASE CONFIGS =====

default_evidence_lm_config = {
    "learning_module_class": EvidenceGraphLM,
    "learning_module_args": {
        # mmd of 0.015 get higher performance but slower run time
        "max_match_distance": 0.01,  # =1cm
        "tolerances": {
            "patch": {
                "principal_curvatures_log": np.ones(2),
                "pose_vectors": np.ones(3) * math.radians(50),
            }
        },
        "feature_weights": {
            # "pose_vectors": np.array([1.0, 0.0, 0.0]), # ignore curvature directions
        },
        # smaller threshold reduces runtime but also performance
        "x_percent_threshold": 20,
        # Using a smaller max_nneighbors (5 instead of 10) makes runtime faster,
        # but reduces performance a bit
        "hypotheses_updater_args": {"max_nneighbors": 10},
        # Use this to update all hypotheses with evidence > 80% of max evidence (faster)
        "evidence_update_threshold": "80%",
        "use_multithreading": False,  # NOTE: could set to True when not debugging
        # NOTE: Currently not used when loading pretrained graphs.
        "max_graph_size": 0.3,  # 30cm
        "num_model_voxels_per_dim": 100,
        "gsg_class": EvidenceGoalStateGenerator,
        "gsg_args": {},
    },
}

num_pretrain_steps = 200

base_ultrasound_experiment = {
    "experiment_class": UltrasoundExperiment,
    "experiment_args": EvalExperimentArgs(
        model_name_or_path=model_path_tbp_robot_lab,
        n_eval_epochs=1,
        max_eval_steps=200,
    ),
    "logging_config": ParallelEvidenceLMLoggingConfig(
        wandb_group="benchmark_experiments",
        # Comment in for quick debugging (turns off wandb and increases logging)
        wandb_handlers=[],
        monty_handlers=[],
        monty_log_level="SILENT",
        python_log_level="DEBUG",
    ),
    "plotting_config": PlottingConfig(
        enabled=False,
        save_path=os.path.join(os.environ["MONTY_DATA"], "ultrasound_test_set/plots"),
        plot_frequency=1,
        plot_patch_features=True,
        show_hypothesis_space=True,
        display_mlh_focus_plot=True,
    ),
    "monty_config": {
        "monty_class": MontyForEvidenceGraphMatching,
        "monty_args": MontyArgs(
            min_eval_steps=20,
            num_exploratory_steps=num_pretrain_steps,
        ),
        "learning_module_configs": {"learning_module_0": default_evidence_lm_config},
        "sensor_module_configs": {
            "sensor_module_0": {
                "sensor_module_class": UltrasoundSM,
                "sensor_module_args": {
                    "sensor_module_id": "patch",
                },
            },
        },
        "motor_system_config": {
            "motor_system_class": MotorSystem,
            "motor_system_args": {
                "policy_class": UltrasoundMotorPolicy,
                "policy_args": make_informed_policy_config(
                    action_space_type="distant_agent_no_translation",
                    action_sampler_class=ConstantSampler,
                    rotation_degrees=1.0,
                    use_goal_state_driven_actions=False,
                ),
            },
        },
        "sm_to_agent_dict": {
            "patch": "agent_id_0",
        },
        "sm_to_lm_matrix": [[0]],
        "lm_to_lm_matrix": None,
        "lm_to_lm_vote_matrix": None,
    },
    "dataset_class": EnvironmentDataset,
    "dataset_args": {
        "env_init_func": UltrasoundEnvironment,
        "env_init_args": {
            "data_path": os.path.join(
                os.environ["MONTY_DATA"], "ultrasound/ultrasound_robot_lab_test/"
            ),
        },
        "transform": None,
    },
    "eval_dataloader_class": UltrasoundDataLoader,
    "eval_dataloader_args": {"patch_size": 256},
}

# ===== SIM-TO-REAL INFERENCE CONFIGS =====

# Experiment for testing offline on a dataset that was collected with the ultrasound
# probe and saved to JSON files. Can be used to experiment without having the whole
# ultrasound and tracking set up and for repeatable experiments.
json_dataset_ultrasound_infer_sim2real = deepcopy(base_ultrasound_experiment)
json_dataset_ultrasound_infer_sim2real["monty_config"]["monty_args"] = MontyArgs(
    min_eval_steps=199,  # Ensure all points are used for inference
    num_exploratory_steps=num_pretrain_steps,
)

# ===== LEARNING ON ULTRASOUND DATA CONFIGS =====

# For learning we use the DisplacementGraphLM.
LM_config_for_learning = {
    "learning_module_0": {
        "learning_module_class": DisplacementGraphLM,
        "learning_module_args": {
            "k": 10,
            "match_attribute": "displacement",
            "tolerance": np.ones(3) * 0.0001,
            "graph_delta_thresholds": {
                "patch": {
                    "distance": 0.001,
                    "pose_vectors": [np.pi / 8, np.pi * 2, np.pi * 2],
                    "curvature": [1.0, 1.0],
                }
            },
        },
    }
}

OBJECT_FOR_LEARNING = "tuna_can"

# Loads an offline .json dataset and trains models on it.
json_dataset_ultrasound_learning = deepcopy(json_dataset_ultrasound_infer_sim2real)
json_dataset_ultrasound_learning.update(
    {
        "experiment_args": EvalExperimentArgs(
            do_train=True,
            do_eval=False,
            n_train_epochs=1,
            max_total_steps=num_pretrain_steps,
            max_train_steps=num_pretrain_steps,
        ),
        "dataset_args": {
            "env_init_func": JSONDatasetUltrasoundEnvironment,
            "env_init_args": {
                "data_path": os.path.join(
                    os.environ["MONTY_DATA"],
                    f"ultrasound/ultrasound_robot_lab_train/{OBJECT_FOR_LEARNING}/",
                ),
            },
        },
        "train_dataloader_class": UltrasoundDataLoader,
        "train_dataloader_args": {"patch_size": 256},
        "plotting_config": PlottingConfig(
            enabled=False,
        ),
    }
)
json_dataset_ultrasound_learning["monty_config"]["learning_module_configs"] = (
    LM_config_for_learning
)
json_dataset_ultrasound_learning["monty_config"]["monty_class"] = MontyForGraphMatching

# Learn with datapoints intended for inference; used to sanity check the quality of
# the collected dataset
json_dataset_ultrasound_learning_inference_data = deepcopy(
    json_dataset_ultrasound_learning
)
json_dataset_ultrasound_learning_inference_data["dataset_args"]["env_init_args"][
    "data_path"
] = os.path.join(
    os.environ["MONTY_DATA"],
    f"ultrasound/ultrasound_robot_lab_train/{OBJECT_FOR_LEARNING}/",
)

# ===== PROBE-TRIGGERED EXPERIMENTS =====

# Define an experiment that is interactively triggered by use of the probe. Any
# observations are saved, forming a potential starting piont for a dataset for a
# particular object. This pipeline is semi-manual, as the operator needs to move
# these observations into a directory appropriate for the particular object.
# Also makes use of goal-state generation. The provided goal states can be visualized
# by the user in the 3D visualization of the phantom setup.

evidence_lm_config_with_gsg = deepcopy(default_evidence_lm_config)
evidence_lm_config_with_gsg["learning_module_args"]["gsg_args"] = {
    "goal_tolerances": {
        "location": 0.015,  # distance in meters
    },  # Tolerance(s) when determining goal-state success
    "elapsed_steps_factor": 5,  # Factor that considers the number of elapsed
    # steps as a possible condition for initiating a hypothesis-testing goal
    # state; should be set to an integer reflecting a number of steps
    "min_post_goal_success_steps": 5,  # Number of necessary steps for a hypothesis
    # goal-state to be considered
    "x_percent_scale_factor": 0.75,  # Scale x-percent threshold to decide
    # when we should focus on pose rather than determining object ID; should
    # be bounded between 0:1.0; "mod" for modifier
    "desired_object_distance": 0.06,  # Distance from the object to the
    # agent that is considered "close enough" to the object
}

try:
    VIVE_SERVER_URL = os.environ.get("VIVE_SERVER_URL")
except Exception as e:
    print(f"Error getting VIVE_SERVER_URL from environment: {e}")
    print(
        "Please set the VIVE_SERVER_URL environment variable, e.g. VIVE_SERVER_URL='http://192.168.1.237:3001'"
    )
    sys.exit(1)

POSE_ENDPOINT = f"http://{VIVE_SERVER_URL}:3001/pose"

base_probe_triggered_experiment = deepcopy(base_ultrasound_experiment)
base_probe_triggered_experiment["dataset_args"]["env_init_func"] = (
    ProbeTriggeredUltrasoundEnvironment
)
base_probe_triggered_experiment["dataset_args"]["env_init_args"] = {
    "image_listen_port": 3000,
    "save_path": os.path.join(os.environ["MONTY_DATA"], "ultrasound_data_collection/"),
    "vive_url": POSE_ENDPOINT,
}
base_probe_triggered_experiment["monty_config"]["learning_module_configs"] = {
    "learning_module_0": evidence_lm_config_with_gsg
}
base_probe_triggered_experiment["monty_config"]["monty_class"] = (
    MontyForEvidenceGraphMatchingWithGoalStateServer
)
base_probe_triggered_experiment["monty_config"]["motor_system_config"] = {
    "motor_system_class": MotorSystem,
    "motor_system_args": {
        "policy_class": UltrasoundMotorPolicy,
        "policy_args": make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=1.0,
            use_goal_state_driven_actions=True,
        ),
    },
}

# Override plotting config to enable plotting for probe triggered experiments
base_probe_triggered_experiment["plotting_config"] = PlottingConfig(
    enabled=True,
    save_path=os.path.join(
        os.environ["MONTY_DATA"], "ultrasound_data_collection/plots"
    ),
    plot_frequency=1,
    plot_patch_features=True,
    show_hypothesis_space=True,
)

# Collect data for offline learning - more eval steps, enabling more points to be
# collected. NOTE that the operator should aim for a systematic exploration of the
# object, as this will be crucial for the quality of the learned model.
LEARNING_DATA_POINTS = 200
probe_triggered_data_collection_for_learning = deepcopy(base_probe_triggered_experiment)
probe_triggered_data_collection_for_learning["monty_config"]["monty_args"] = MontyArgs(
    min_eval_steps=LEARNING_DATA_POINTS,
    max_total_steps=LEARNING_DATA_POINTS + 1,
    num_exploratory_steps=num_pretrain_steps,
)
probe_triggered_data_collection_for_learning["experiment_args"] = EvalExperimentArgs(
    model_name_or_path=model_path_tbp_robot_lab,
    n_eval_epochs=1,
    max_eval_steps=LEARNING_DATA_POINTS,
)

# Collect data for offline inference - fewer eval steps, since we are interested
# in the ability to perform inference given a minimal exploration of the object.
# NOTE that the operator should aim for a rapid exploration of the object, e.g.
# frequently moving from one side of an object to the other, and making sure to include
# features that would distinguish different objects.
INFERENCE_DATA_POINTS = 50
probe_triggered_data_collection_for_inference = deepcopy(
    base_probe_triggered_experiment
)
probe_triggered_data_collection_for_inference["monty_config"]["monty_args"] = MontyArgs(
    min_eval_steps=INFERENCE_DATA_POINTS,
    max_total_steps=INFERENCE_DATA_POINTS + 1,
    num_exploratory_steps=num_pretrain_steps,
)
probe_triggered_data_collection_for_inference["experiment_args"] = EvalExperimentArgs(
    model_name_or_path=model_path_tbp_robot_lab,
    n_eval_epochs=1,
    max_eval_steps=INFERENCE_DATA_POINTS,
)



CONFIGS = {
    "json_dataset_ultrasound_infer_sim2real": json_dataset_ultrasound_infer_sim2real,
    "json_dataset_ultrasound_learning": json_dataset_ultrasound_learning,
    "json_dataset_ultrasound_learning_inference_data": json_dataset_ultrasound_learning_inference_data,
    "probe_triggered_data_collection_for_learning": probe_triggered_data_collection_for_learning,
    "probe_triggered_data_collection_for_inference": probe_triggered_data_collection_for_inference,
}
