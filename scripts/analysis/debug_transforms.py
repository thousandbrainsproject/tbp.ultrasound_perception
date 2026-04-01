import os

import matplotlib.pyplot as plt
import numpy as np
from tbp.monty.frameworks.utils.logging_utils import load_stats

LOAD_PRETRAINED_MODEL = True
LOAD_LOGGING_DATA = True


def get_patch_world_location(
    agent_position,
    sensor_position,
    agent_rotation,
    image_depth,
):
    """Calculates the patch's location and orientation in world coordinates.
    current version:
    # adding the 0.03
    offset_direction = np.array([0.0, 0.0, 1.0])
    rotated_offset_direction = agent_rotation @ offset_direction
    offset_distance = sensor_position[1]  # = 0.03
    relative_offset = offset_distance * rotated_offset_direction
    agent_position = agent_position + relative_offset

    # adding the 0.09 + depth
    fake_sensor_rel_world = agent_rotation @ np.array([0.0, 1.0, 0.0])
    offset_distance = sensor_position[2] + image_depth

    # Determine the movement vector.
    # The vector is in the opposite direction of the unit_normal (-unit_normal)
    # and scaled by the offset_distance.
    movement_vector = -fake_sensor_rel_world * offset_distance

    # Compute the new location by applying the movement_vector to the agent's current position.
    patch_world_location = agent_position + movement_vector

    return patch_world_location
    """
    offset_direction = np.array([0.0, 0.0, 1.0])
    rotated_offset_direction = agent_rotation @ offset_direction
    offset_distance = sensor_position[1]  # = 0.03
    relative_offset = offset_distance * rotated_offset_direction
    agent_position = agent_position + relative_offset

    fake_sensor_rel_world = agent_rotation @ np.array([0.0, 1.0, 0.0])
    offset_distance = sensor_position[2] + image_depth

    # Determine the movement vector.
    # The vector is in the opposite direction of the unit_normal (-unit_normal)
    # and scaled by the offset_distance.
    movement_vector = -fake_sensor_rel_world * offset_distance

    # Compute the new location by applying the movement_vector to the agent's current position.
    patch_world_location = agent_position + movement_vector

    return patch_world_location


if LOAD_PRETRAINED_MODEL:
    # load pretrained model
    pretrain_path = os.path.expanduser("~/tbp/results/monty/pretrained_models/")
    pretrained_dict = (
        pretrain_path + "ultrasound_robot_lab_v1/surf_agent_1lm_tbp_robot_lab/pretrained/"
    )

    log_path = os.path.expanduser(
        "~/tbp/results/monty/projects/evidence_eval_runs/logs/"
    )
    exp_name = "json_dataset_ultrasound_learning"
    exp_path = log_path + exp_name

    train_stats, eval_stats, detailed_stats, lm_models = load_stats(
        exp_path,
        load_train=False,  # doesn't load train csv
        load_eval=False,  # loads eval_stats.csv
        load_detailed=False,  # doesn't load .json
        load_models=True,  # loads .pt models
        pretrained_dict=pretrained_dict,
    )
    # model = lm_models["0"]["LM_0"]["new_object0"]["patch"]
    model = lm_models["pretrained"][0]["potted_meat_can"]["patch"]
    model_locs = np.array(model.pos)
    model_normals = np.array(model.norm)

if LOAD_LOGGING_DATA:
    logging_data = np.load("logging_data.npz")
    tracker_positions = logging_data["tracker_position"]
    probe_positions = logging_data["probe_position"]
    tracker_orientations = logging_data["tracker_orientation"]
    image_depths = logging_data["max_probe_depth"]
    normals_rel_sensor = logging_data["normal_rel_sensor"]

    patch_rel_world = []
    sensor_rel_world = []
    sensor_rel_world_2 = []
    for i in range(len(tracker_positions)):
        sensor_rel_world.append(
            get_patch_world_location(
                tracker_positions[i],
                probe_positions[i],
                tracker_orientations[i],
                0.0,
            )
        )
        probe_y_position = np.zeros(3)
        probe_y_position[2] = probe_positions[i][2]
        sensor_rel_world_2.append(
            get_patch_world_location(
                tracker_positions[i],
                probe_y_position,
                tracker_orientations[i],
                image_depths[i],
            )
        )
        patch_rel_world.append(
            get_patch_world_location(
                tracker_positions[i],
                probe_positions[i],
                tracker_orientations[i],
                image_depths[i],
            )
        )
    locs_to_plot = np.array(patch_rel_world)

    normals_rel_world = []
    for i in range(len(tracker_orientations)):
        normals_rel_world.append(tracker_orientations[i] @ (-normals_rel_sensor[i]))
    normals_to_plot = np.array(normals_rel_world)


def plot_points_and_normals(ax, locs, normals, pn_color="red"):
    """Plots points and their normals on a given 3D axes."""
    colors = np.linspace(0, 1, len(locs))
    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], c=colors)
    # add point normals to plot
    ax.quiver(
        locs[:, 0],
        locs[:, 1],
        locs[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
        length=0.02,
        color=pn_color,
    )


def center_locations(locs):
    """center np array of locations on the origin"""
    return locs - np.mean(locs, axis=0)


# Create the figure and 3D axes once
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
# plot_points_and_normals(ax, tracker_positions, normals_to_plot, pn_color="red")
plot_points_and_normals(
    ax, center_locations(np.array(sensor_rel_world_2)), normals_to_plot, pn_color="blue"
)
# plot_points_and_normals(
#     ax, np.array(sensor_rel_world), normals_to_plot, pn_color="orange"
# )
# plot_points_and_normals(ax, locs_to_plot, normals_to_plot, pn_color="green")

plot_points_and_normals(
    ax, center_locations(model_locs), model_normals, pn_color="green"
)

# Set up plot properties and show
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.tight_layout()
plt.show()
