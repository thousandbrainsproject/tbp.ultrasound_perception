# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy

import numpy as np
from scipy.spatial.transform import Rotation
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
)
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
)
from tbp.monty.frameworks.utils.transform_utils import scipy_to_numpy_quat


class UltrasoundDataLoader(EnvironmentDataLoader):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.episode_counter = 0

    def __iter__(self):
        # Reset the environment before iterating
        self._observation, proprioceptive_state = self.dataset.reset()
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        self._action = None
        self._counter = 0
        return self

    def __next__(self):
        self._observation, proprioceptive_state = self.dataset[None]
        if self._observation is None:
            raise StopIteration
        self.motor_system._state = (
            MotorSystemState(proprioceptive_state) if proprioceptive_state else None
        )
        full_observation = copy.deepcopy(self._observation)
        patch_observation = self.extract_patch(full_observation)
        patch_observation = self.add_proprioceptive_state(
            patch_observation, proprioceptive_state
        )
        return patch_observation

    def add_proprioceptive_state(self, observation, proprioceptive_state):
        observation["agent_id_0"]["patch"]["proprioceptive_state_patch"] = (
            proprioceptive_state["agent_id_0"]["sensors"]["ultrasound"]
        )
        observation["agent_id_0"]["patch"]["proprioceptive_state_agent"] = {}
        observation["agent_id_0"]["patch"]["proprioceptive_state_agent"]["position"] = (
            proprioceptive_state["agent_id_0"]["position"]
        )
        observation["agent_id_0"]["patch"]["proprioceptive_state_agent"]["rotation"] = (
            proprioceptive_state["agent_id_0"]["rotation"]
        )
        return observation

    def extract_patch(self, observation):
        full_image = np.array(observation["agent_id_0"]["ultrasound"]["img"])
        patch, patch_pixel_start = self.find_patch_with_highest_gradient(
            full_image,
            patch_size=self.patch_size,
        )
        observation["agent_id_0"]["patch"] = {}
        observation["agent_id_0"]["patch"]["img"] = patch
        observation["agent_id_0"]["patch"]["patch_pixel_start"] = patch_pixel_start
        observation["agent_id_0"]["patch"]["full_image_height"] = full_image.shape[0]
        # remove the ultrasound image from the observation
        del observation["agent_id_0"]["ultrasound"]
        return observation

    def find_patch_with_highest_gradient(
        self, full_image, patch_size, grid_size=9, window_size=100
    ):
        """Finds the first patch with a significant horizontal edge in the ultrasound image.

        Takes a patch at the center top of the image and shifts it down in the middle
        column. Each time it calculates the horizontal edges using a Sobel filter on a
        9x9 grid of mean intensities on a 0.5 sized patch. Based on the center location of
        the first patch that shows a significant local peak it extracts a patch of size
        patch_size x patch_size.

        Args:
            full_image (np.ndarray): The full ultrasound image of shape (N, M)
            patch_size (int): Size of the square patch to extract
            grid_size (int): Number of bins to group the pixel values into along each
                dimension of the patch for calculating the mean intensity.
            window_size (int): Size of the window to calculate the local mean and std
                of the gradient.
        Returns:
            tuple: (patch, patch_pixel_start) where:
                - patch (np.ndarray): The first patch with significant horizontal edge
                - patch_pixel_start (int): the y pixel coordinate of start of the
                    patch; e.g. if the image is 200x800 pixels, this might be
                    pixel 240
        """
        height, width = full_image.shape
        x_center = width // 2
        start_y = patch_size // 2

        best_location = None
        y_starting_positions = []
        y_central_positions = []
        gradients = []

        # Define Sobel kernel for horizontal edge detection
        # TODO: check this is the best kernel for what we want
        sobel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Collect all gradients by moving the patch down the center column
        # Use a smaller patch for this
        test_patch_size = patch_size // 2
        for y in range(start_y, height - patch_size // 2):
            # Extract patch
            patch = full_image[
                y - test_patch_size // 2 : y + test_patch_size // 2,
                x_center - test_patch_size // 2 : x_center + test_patch_size // 2,
            ]

            cell_size = test_patch_size // grid_size
            # Compute mean intensity for each cell
            cell_means = np.zeros((grid_size, grid_size))
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = patch[
                        i * cell_size : (i + 1) * cell_size,
                        j * cell_size : (j + 1) * cell_size,
                    ]
                    cell_means[i, j] = np.mean(cell)

            # Pad the cell_means for Sobel filtering
            padded_means = np.pad(cell_means, 1, mode="edge")

            # Apply Sobel filter to detect horizontal edges
            edge_response = np.zeros_like(cell_means)
            for i in range(grid_size):
                for j in range(grid_size):
                    # Extract 3x3 region for convolution
                    region = padded_means[i : i + 3, j : j + 3]
                    # Apply Sobel filter
                    edge_response[i, j] = np.sum(region * sobel_horizontal)

            # Calculate total edge response
            total_gradient = np.sum(np.abs(edge_response))

            # Note we use the initial y position of the patch for the depth,
            # as later we will determine the location of the edge within the patch for
            # the final depth reading; note this should use the patch_size, not the
            # test_patch_size, as it defines the starting y of the patch that we 
            # will return
            starting_y = y - patch_size // 2
            y_starting_positions.append(starting_y)
            y_central_positions.append(y)
            gradients.append(total_gradient)

        gradients = np.array(gradients)
        y_starting_positions = np.array(y_starting_positions)
        y_central_positions = np.array(y_central_positions)
        max_gradient = np.max(gradients)

        # Pad the gradients array to allow local window calculations for earlier values
        padded_gradients = np.pad(gradients, window_size, mode="edge")

        # Find the first significant local peak
        for i in range(len(gradients)):
            # Get local window from padded array
            local_window = padded_gradients[i : i + 2 * window_size]
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            local_threshold = local_mean + local_std

            # Check if current point is a peak and above threshold
            # Also ensure local std isn't too small (indicating a flat region)
            if (
                gradients[i] > gradients[max(0, i - 1)]
                and gradients[i] > gradients[min(len(gradients) - 1, i + 1)]
                and gradients[i] > local_threshold
                and local_std > (np.std(gradients) // 10)
                and gradients[i] > max_gradient / 2
            ):
                best_central_location = (y_central_positions[i], x_center)
                best_starting_location = (y_starting_positions[i], x_center)
                break

        # If no peak found, use the maximum gradient
        if best_location is None:
            max_idx = np.argmax(gradients)
            best_central_location = (y_central_positions[max_idx], x_center)
            best_starting_location = (y_starting_positions[max_idx], x_center)

        # Extract the final patch at the selected location
        y, x = best_central_location
        best_patch = full_image[
            y - patch_size // 2 : y + patch_size // 2,
            x - patch_size // 2 : x + patch_size // 2,
        ]

        y_start, x_start = best_starting_location

        # # Visualize the results
        # plt.figure(figsize=(15, 5))

        # # Plot full image
        # plt.subplot(131)
        # plt.imshow(full_image, cmap="gray")
        # plt.title("Full Ultrasound Image")

        # # Draw rectangle around the selected patch
        # rect = plt.Rectangle(
        #     (x - patch_size // 2, y - patch_size // 2),
        #     patch_size,
        #     patch_size,
        #     fill=False,
        #     color="red",
        #     linewidth=2,
        # )
        # plt.gca().add_patch(rect)

        # # Plot the selected patch
        # plt.subplot(132)
        # plt.imshow(best_patch, cmap="gray")
        # plt.title("Selected Patch")

        # # Plot gradient curve
        # plt.subplot(133)
        # plt.plot(y_positions, gradients, "b-", label="Edge Response")

        # # Calculate and plot local threshold for all points
        # local_thresholds = []
        # local_stds = []
        # for i in range(len(gradients)):
        #     local_window = padded_gradients[i : i + 2 * window_size]
        #     local_mean = np.mean(local_window)
        #     local_std = np.std(local_window)
        #     local_thresholds.append(local_mean + local_std)
        #     local_stds.append(local_std)

        # plt.plot(
        #     y_positions, local_thresholds, "g--", label="Local Threshold (mean + std)"
        # )
        # plt.plot(y_positions, local_stds, "r--", label="Local Std")
        # plt.axhline(
        #     y=(np.std(gradients) // 10),
        #     color="k",
        #     linestyle=":",
        #     label="Min Std Threshold",
        # )

        # # Plot selected position
        # plt.axvline(
        #     x=best_location[0], color="r", linestyle="--", label="Selected Position"
        # )

        # # Add text to indicate if we used max gradient
        # if best_location is not None:
        #     i = np.where(y_positions == best_location[0])[0][0]
        #     local_window = padded_gradients[i : i + 2 * window_size]
        #     local_mean = np.mean(local_window)
        #     local_std = np.std(local_window)
        #     local_threshold = local_mean + local_std
        #     if gradients[i] <= local_threshold or local_std <= (
        #         np.std(gradients) // 10
        #     ):
        #         plt.text(
        #             0.05,
        #             0.95,
        #             "Using max gradient (no significant peak found)",
        #             transform=plt.gca().transAxes,
        #             color="red",
        #             bbox=dict(facecolor="white", alpha=0.8),
        #         )

        # plt.xlabel("Y Position")
        # plt.ylabel("Horizontal Edge Response")
        # plt.title("Horizontal Edge Response vs Y Position")
        # plt.legend()
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()

        return best_patch, y_start

    def change_object_by_idx(self, idx):
        # NOTE: We don't have ground truth rotation for the object so just use 0.
        euler_rotation = np.zeros(3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        self.primary_target = {
            "object": self.dataset.env.object_names[idx],
            "semantic_id": 0,
            "rotation": quat_rotation,
            "euler_rotation": euler_rotation,
            "quat_rotation": q,
            "position": np.zeros(3),
            "scale": np.ones(3),
        }

    def pre_epoch(self):
        self.change_object_by_idx(idx=self.episode_counter)

    def post_episode(self):
        """
        Call the environment to update the "scene" (load the next folder with the
        next scanned object), while updating the corresponding primary target
        via the change_object_by_idx method.
        """
        print("In post episode, switching to next scene")
        print(f"Current scene: {self.dataset.env.current_scene}")
        print(f"Scene names: {self.dataset.env.object_names}")
        self.episode_counter += 1
        self.dataset.env.switch_to_next_scene()
        print(f"\n\nNew current scene: {self.dataset.env.current_scene}")
        print("Corresponding primary target: ", self.primary_target)
        print(f"Episode counter: {self.episode_counter}")

        if self.episode_counter < len(self.dataset.env.object_names):
            self.change_object_by_idx(idx=self.episode_counter)
        else:
            print("Reached the end of the dataset. Stopping episode.")