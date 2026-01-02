# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import json
import os
import time

import numpy as np
import PIL
import quaternion as qt
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_environment import ActionSpace
from tbp.monty.frameworks.environments.two_d_data import (
    EmbodiedEnvironment,
)


class UltrasoundActionSpace(tuple, ActionSpace):
    """Action space placeholder (Monty doesn't act here)."""

    def sample(self):
        return self.rng.choice(self)


class UltrasoundEnvironment(EmbodiedEnvironment):
    """Base Ultrasound Environment.

    NOTE: This is not fully functional (doesn't retrieve probe pose from trackers). Use
    JSONDatasetUltrasoundEnvironment or ProbeTriggeredUltrasoundEnvironment for
    actual data loading and state retrieval.
    """
    def __init__(self, data_path=None):
        """Initialize environment.

        Args:
            patch_size: height and width of patch in pixels, defaults to 64
            data_path: path to the image dataset. If None its set to
                ~/tbp/data/ultrasound/ultrasound_stream/
        """
        self.data_path = data_path
        self.full_image = None  # Store the full image for plotting

        # Scene names used to load the next ultrasound image
        self.scene_names = sorted(
            [a for a in os.listdir(self.data_path) if a[0] != "."]
        )

        # Strip the number from the scene names such that 002_montys_brain becomes montys_brain, etc.
        self.object_names = ["_".join(name.split("_")[1:]) for name in self.scene_names]

        self.current_scene = 0
        self.step_count = 0

        # Just for compatibility.
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]
        self._valid_actions = ["next"]

    @property
    def action_space(self):
        return UltrasoundActionSpace(["next"])

    def step(self, action: Action):
        """Retrieve the next observation.

        Args:
            action: load the next image + tracking data.

        Returns:
            observation (dict).
        """
        self.current_ultrasound_image = self.load_next_ultrasound_image()

        obs = {
            "agent_id_0": {
                "ultrasound": {
                    "img": self.current_ultrasound_image,
                },
            }
        }
        self.step_count += 1
        return obs

    def load_next_ultrasound_image(self):
        current_img_path = (
            self.data_path
            + f"{self.scene_names[self.current_scene]}/img_{self.step_count}.png"
        )
        print(f"Looking for ultrasound image from {current_img_path}")
        # Load ultrasound image
        wait_count = 0
        while not os.path.exists(current_img_path):
            if wait_count % 10 == 0:
                # Print every 10 seconds
                print("Waiting for new ultrasound data...")
            time.sleep(1)
            wait_count += 1

        load_succeeded = False
        while not load_succeeded:
            try:
                current_ultrasound_image = self.load_ultrasound_image(current_img_path)
                load_succeeded = True
            except PIL.UnidentifiedImageError:
                print("waiting for rgb file to finish streaming")
                time.sleep(1)

        return current_ultrasound_image

    def load_ultrasound_image(self, img_path):
        """Load RGB image and convert to grayscale.

        Returns:
            np.ndarray: The grayscale image of shape (M, N)
        """
        rgb_image = np.array(PIL.Image.open(img_path))
        # Convert to grayscale by taking mean across color channels
        grayscale_image = np.mean(rgb_image, axis=2)
        self.full_image = grayscale_image
        return grayscale_image

    def get_state(self):
        """Get agent state.

        Returns:
            The agent state.
        """
        # NOTE: these are just placeholders. In the JSONDatasetUltrasoundEnvironment and
        # ProbeTriggeredUltrasoundEnvironment, the state actual probe pose is retrieved
        # from the trackers.
        agent_position = np.array([0, 0, 0])
        agent_rotation = qt.quaternion(1, 0, 0, 0)

        state = {
            "agent_id_0": {
                # Patch pose is placeholder and will be determined in SM
                "sensors": {
                    "ultrasound": {
                        "rotation": qt.quaternion(1, 0, 0, 0),
                        "position": np.array([0, 0, 0]),
                    },
                },
                "rotation": agent_rotation,
                "position": agent_position,
            }
        }
        return state

    def reset(self):
        """Reset environment and extract image patch.

        Returns:
            The observation from the image patch.
        """
        first_obs = self.step(None)
        self.step_count = 0
        return first_obs

    def switch_to_next_scene(self):
        self.current_scene += 1
        self.step_count = 0

    def add_object(self, *args, **kwargs):
        raise NotImplementedError(
            "UltrasoundEnvironment does not support adding objects"
        )

    def remove_all_objects(self):
        raise NotImplementedError(
            "UltrasoundEnvironment does not support removing all objects"
        )

    def close(self):
        self._current_state = None

    def get_full_image(self):
        """Get the current full ultrasound image.

        Returns:
            np.ndarray: The complete ultrasound image
        """
        if self.full_image is None:
            print("Warning: full_image is None")
            return np.zeros(
                (256, 256), dtype=np.float32
            )  # Return empty image as fallback
        return self.full_image


class JSONDatasetUltrasoundEnvironment(UltrasoundEnvironment):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def step(self, action: Action):
        """Retrieve the next observation.

        Args:
            action: unused. We just load the next data point in the dataset.

        Returns:
            observation (dict).
        """
        self.current_ultrasound_image, self.current_state = self.load_next_data_point()
        if self.current_ultrasound_image is None:
            return None

        obs = self.current_ultrasound_image
        self.step_count += 1
        return obs

    def load_next_data_point(self):
        """Load the next ultrasound image from the dataset."""
        try:
            with open(
                os.path.join(
                    self.data_path,
                    f"{self.scene_names[self.current_scene]}",
                    f"{self.step_count}.json",
                ),
                "r",
            ) as f:
                data = json.load(f)
        except FileNotFoundError:
            # This will end the episode
            return None, None

        self.full_image = np.array(data["obs"]["agent_id_0"]["ultrasound"]["img"])
        # Overwrite the position of the probe.
        # TODO: can remove this now that it is added to the ProbeTriggeredUltrasoundEnvironment
        # (If used during data collection)
        # TODO: this just looked like it gave the best results but it is not ideal yet.
        data["state"]["agent_id_0"]["sensors"]["ultrasound"]["position"] = np.array(
            [0, 0.028, 0.105]
        )
        return data["obs"], data["state"]

    def get_state(self):
        return self.current_state
