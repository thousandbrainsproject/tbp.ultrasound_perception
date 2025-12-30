import copy
import json
import os
from typing import Any, Dict
import sys

import cv2
import numpy as np
import quaternion as qt
import requests
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.buffer import BufferEncoder

from custom_classes.environment import UltrasoundEnvironment
from custom_classes.server import ImageServer

try:
    VIVE_SERVER_URL = os.environ.get("VIVE_SERVER_URL")
except Exception as e:
    print(f"Error getting VIVE_SERVER_URL from environment: {e}")
    print(
        "Please set the VIVE_SERVER_URL environment variable, e.g. VIVE_SERVER_URL='http://192.168.1.237:3001'"
    )
    sys.exit(1)

POSE_ENDPOINT = f"http://{VIVE_SERVER_URL}:3001/pose"

class ProbeTriggeredUltrasoundEnvironment(UltrasoundEnvironment):
    def __init__(
        self,
        image_listen_port: int = 8000,
        vive_url: str = "http://localhost:3001/pose",
        save_path: str = None,
    ):
        super().__init__(data_path=None)
        self.image_listen_port = image_listen_port
        self.server = ImageServer()
        self.server.start(port=image_listen_port)
        self.vive_url = vive_url
        self.vive_pose = None
        self.save_path = save_path

        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def step(self, action: Action) -> Dict[str, Any]:
        complete_data = False
        while not complete_data:
            self.vive_pose = None
            current_ultrasound_image, metadata = self.server.get_next_image()
            self.full_image = current_ultrasound_image
            self.vive_pose = self.get_vive_pose(metadata["epoch"])
            if self.vive_pose is not None:
                complete_data = True
            else:
                print(
                    "Did not get vive pose data, please ensure the vive service is running and then take another image"
                )
                continue

        obs = {
            "agent_id_0": {
                "ultrasound": {
                    "img": current_ultrasound_image,
                    "metadata": metadata,
                },
            }
        }
        if self.save_path is not None:
            self.save_data(obs, self.get_state())
        self.step_count += 1
        print(
            f"New image detected Obs: ###############################\n{obs}\n########################################################"
        )
        return obs

    def get_state(self):
        """Get agent state.

        Returns:
            The agent state.
        """
        pos = self.vive_pose["pose"]["position"]
        agent_position = np.array([pos["x"], pos["y"], pos["z"]])

        rot = self.vive_pose["pose"]["rotation"]
        agent_rotation = qt.quaternion(rot["w"], rot["x"], rot["y"], rot["z"])

        state = {
            "agent_id_0": {
                "sensors": {
                    "ultrasound": {
                        "rotation": qt.quaternion(1, 0, 0, 0),  # Identity quaternion
                        "position": np.array([0, 0.028, 0.105]),
                    },
                },
                "rotation": agent_rotation,
                "position": agent_position,
            }
        }
        return state

    def save_data(self, obs, state):
        """Combines the two dictionaries into a single dictionary and saves it to a file.
        Saves it to self.save_path named as the current step as .json.
        The image will be saved as a png in the same directory and the .json will contain a reference to it.
        """

        data = {
            "obs": copy.deepcopy(obs),
            "state": state,
        }
        with open(os.path.join(self.save_path, f"{self.step_count}.json"), "w") as f:
            json.dump(data, f, cls=BufferEncoder)

    def close(self):
        super().close()

    def get_vive_pose(self, epoch: float) -> Dict[str, Any]:
        try:
            print(f"Getting vive pose for epoch: {epoch}")
            response = requests.get(f"{self.vive_url}?epoch={epoch}")
            print(f"Pose response: {response.json()}")
            if not 200 <= response.status_code < 300:
                return None
            return response.json()["data"]
        except Exception:
            return None
