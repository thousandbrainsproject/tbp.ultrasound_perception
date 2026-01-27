#!/usr/bin/env python3
"""
Pose logger for a VIVE Tracker 3.0 (no HMD).
Outputs position (metres) and orientation (yaw,-pitch,-roll in °, SteamVR Y-X-Z order).
Includes VPython visualization of the tracker's pose.
Connects to HTTP service instead of using OpenVR directly.
"""
import json
import os
import sys
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation
from vpython import arrow, box, button, canvas, color, mag, rate, vector

# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
VIVE_SERVER_URL = os.environ.get("VIVE_SERVER_URL")
if VIVE_SERVER_URL is None:
    print(f"Error getting VIVE_SERVER_URL from environment.")
    print(
        "Please set the VIVE_SERVER_URL environment variable, e.g. VIVE_SERVER_URL='http://192.168.1.237:3001'"
    )
    sys.exit(1)
else:
    POSE_ENDPOINT = f"http://{VIVE_SERVER_URL}:3001/pose"

# Global variable to store the latest goal state
latest_goal_state = None

# Global variable to store reference to set_external_reference_pose function
set_external_reference_callback = None

# HTTP Server for receiving goal state
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class GoalStateHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global latest_goal_state, set_external_reference_callback
        if self.path == "/goal_state":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                latest_goal_state = data.get("goal_state")
                # For now, we just print it. Visualization will be added later.
                print(f"Received goal state: {latest_goal_state}")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Goal state received")
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON")
        elif self.path == "/setexternal":
            try:
                if set_external_reference_callback:
                    set_external_reference_callback()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"External reference pose set")
                    print("External reference pose set via HTTP request")
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Set external reference function not available")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error setting external reference: {str(e)}".encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_goal_state_server(port=3003):
    server_address = ("", port)  # Listen on all available interfaces
    httpd = HTTPServer(server_address, GoalStateHandler)
    print(f"Goal state server running on port {port}")
    httpd.serve_forever()

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def fetch_pose_from_server():
    """Fetch pose data from the HTTP service."""
    try:
        response = requests.get(POSE_ENDPOINT, timeout=0.1)
        if response.status_code == 200:
            data = response.json()
            return data.get("data"), True
        else:
            return None, False
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None, False


def format_pose(pose_data):
    """Convert pose data from HTTP service to position vector and rotation object."""
    if not pose_data or "pose" not in pose_data:
        return None, None

    pose = pose_data["pose"]
    position_data = pose["position"]
    rotation_data = pose["rotation"]

    # Position
    position = vector(position_data["x"], position_data["y"], position_data["z"])

    # Rotation (convert from w,x,y,z to scipy format)
    quat_wxyz = [
        rotation_data["w"],
        rotation_data["x"],
        rotation_data["y"],
        rotation_data["z"],
    ]
    quat_xyzw = [
        quat_wxyz[1],
        quat_wxyz[2],
        quat_wxyz[3],
        quat_wxyz[0],
    ]  # Convert to scipy format
    r = Rotation.from_quat(quat_xyzw)

    return position, r


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def track_pose():
    # Stores the latest pose data for the button callback
    latest_pose_data = None

    # Start the goal state server in a separate thread
    goal_server_thread = threading.Thread(target=run_goal_state_server, args=(3003,))
    goal_server_thread.daemon = True  # Daemonize thread to exit when main thread exits
    goal_server_thread.start()

    external_ref_x_arrow = None
    external_ref_y_arrow = None
    external_ref_z_arrow = None
    origin_box = None  # For the reference box/cereal box
    goal_state_arrow = None  # For visualizing the goal state
    # Length for the external reference axes arrows
    EXTERNAL_REF_ARROW_LENGTH = 0.03
    PROBE_ARROW_LENGTH = 0.05
    GOAL_ARROW_LENGTH = 0.1  # Length for the goal state arrow
    GOAL_PROXIMITY_THRESHOLD = 0.04  # Distance in meters to consider goal reached
    # Dimensions for the reference "cereal box"
    CEREAL_BOX_LENGTH = 0.25
    CEREAL_BOX_HEIGHT = 0.25
    CEREAL_BOX_WIDTH = 0.1

    # Camera control constants for side-on view of the reference box
    CAMERA_DISTANCE_FROM_BOX_SIDE = 0.5  # Distance from box center along its Y-axis

    # --- Define initial camera parameters for the scene's first view ---
    _initial_cam_pos_world = vector(0.0, 0.55, -0.4)
    _initial_cam_axis_world = vector(0.0, -0.2, 1.5)

    _approx_start_loc_val = vector(0, 0.5, 0)

    # Define conceptual fixed positions for light sources in world coordinates.
    # These determine the direction from which light comes to illuminate the origin_box.
    LIGHT_SOURCE_POS_0 = vector(0.5, 0.5, 1.0)
    LIGHT_SOURCE_POS_1 = vector(-0.5, 0.5, -1.0)

    def set_external_reference_pose():
        nonlocal latest_pose_data
        nonlocal external_ref_x_arrow, external_ref_y_arrow, external_ref_z_arrow
        nonlocal origin_box
        nonlocal CEREAL_BOX_LENGTH, CEREAL_BOX_HEIGHT

        if latest_pose_data:
            reference_pos, pose_r_obj = format_pose(latest_pose_data)

            if reference_pos is None or pose_r_obj is None:
                print("Cannot set external reference: invalid pose data.")
                return

            if external_ref_x_arrow:  # Check if initialized
                external_ref_x_arrow.pos = reference_pos
                external_ref_x_arrow.axis = vector(
                    *pose_r_obj.apply((-EXTERNAL_REF_ARROW_LENGTH, 0, 0))
                )
            if external_ref_y_arrow:
                external_ref_y_arrow.pos = reference_pos
                external_ref_y_arrow.axis = vector(
                    *pose_r_obj.apply((0, -EXTERNAL_REF_ARROW_LENGTH, 0))
                )
            if external_ref_z_arrow:
                external_ref_z_arrow.pos = reference_pos
                external_ref_z_arrow.axis = vector(
                    *pose_r_obj.apply((0, 0, EXTERNAL_REF_ARROW_LENGTH))
                )

            if origin_box:
                half_height_offset = vector(
                    *pose_r_obj.apply((0, 0, CEREAL_BOX_HEIGHT / 2))
                )
                origin_box.pos = reference_pos + half_height_offset

                rotated_box_local_x_axis = vector(*pose_r_obj.apply((1, 0, 0)))
                rotated_box_local_z_axis = vector(*pose_r_obj.apply((0, 0, 1)))

                origin_box.axis = rotated_box_local_x_axis * CEREAL_BOX_LENGTH
                origin_box.up = rotated_box_local_z_axis * CEREAL_BOX_HEIGHT

                # Camera update logic for side-on view
                current_box_center_world = origin_box.pos

                camera_target_point_world = current_box_center_world

                # Define camera position:
                scene.camera.pos = (
                    camera_target_point_world
                    + CAMERA_DISTANCE_FROM_BOX_SIDE
                    * ((-1) * (external_ref_x_arrow.axis).norm())
                )

                # Set camera axis (direction it's looking) to point from new position to the target
                scene.camera.axis = camera_target_point_world - scene.camera.pos

                # Update light directions to point towards the origin_box
                # The light.direction is the vector FROM the light source.
                # So, direction = LIGHT_SOURCE_POS - target_box_center.
                if scene.lights:  # scene.lights is a list
                    if len(scene.lights) > 0:  # Check if light exists
                        scene.lights[0].direction = (
                            LIGHT_SOURCE_POS_0 - current_box_center_world
                        )
                    if len(scene.lights) > 1:  # Check if light exists
                        scene.lights[1].direction = (
                            LIGHT_SOURCE_POS_1 - current_box_center_world
                        )

            print(
                "External reference pose updated based on current tracker pose. Camera repositioned and lighting adjusted."
            )
        else:
            print("Cannot set external reference: no pose data available.")

    # Make the function available to the HTTP handler
    global set_external_reference_callback
    set_external_reference_callback = set_external_reference_pose

    try:
        # # Test connection to the HTTP service
        # print(f"Connecting to VIVE tracker service at {VIVE_SERVER_URL}...")
        # test_data, success = fetch_pose_from_server()
        # if not success:
        #     print(f"Cannot connect to VIVE tracker service at {VIVE_SERVER_URL}")
        #     print("Make sure the server.py is running on port 3001")
        #     return

        # if test_data and "serial_number" in test_data:
        #     print(f"Connected to tracker (serial {test_data['serial_number']})")
        # else:
        #     print("Connected to VIVE tracker service")

        # VPython setup
        scene = canvas(title="VIVE Tracker Visualization", width=600, height=600)
        scene.camera.pos = _initial_cam_pos_world  # Camera position (x,y,z)
        scene.camera.axis = (
            _initial_cam_axis_world  # Camera looks towards this direction vector
        )
        # Use the defined light source positions for initial setup.
        # This makes them point towards the world origin (0,0,0) initially.
        # If the origin_box starts far from (0,0,0), the set_external_reference_pose will adjust them later.
        if scene.lights:  # Default lights are usually present
            if len(scene.lights) > 0:
                scene.lights[0].direction = LIGHT_SOURCE_POS_0
            if len(scene.lights) > 1:
                scene.lights[1].direction = LIGHT_SOURCE_POS_1

        # Button to set the current tracker pose as the new origin
        button(
            bind=set_external_reference_pose,
            text="Set current tracker pose as position of external reference",
        )
        scene.append_to_caption("\n")  # Add some space for layout

        approximate_starting_location = _approx_start_loc_val
        origin_box = box(
            pos=approximate_starting_location + vector(0, CEREAL_BOX_HEIGHT / 2, 0),
            size=vector(CEREAL_BOX_LENGTH, CEREAL_BOX_HEIGHT, CEREAL_BOX_WIDTH),
            color=color.gray(0.5),
            opacity=0.7,
        )

        # External reference in the form of a "cereal box" object
        external_ref_x_arrow = arrow(
            pos=vector(0, 0, 0) + approximate_starting_location,
            length=EXTERNAL_REF_ARROW_LENGTH,
            shaftwidth=0.005,
            color=color.red,
            axis=vector(-1, 0, 0),
            opacity=0.5,
        )
        external_ref_y_arrow = arrow(
            pos=vector(0, 0, 0) + approximate_starting_location,
            length=EXTERNAL_REF_ARROW_LENGTH,
            shaftwidth=0.005,
            color=color.blue,
            axis=vector(0, -1, 0),
            opacity=0.5,
        )
        external_ref_z_arrow = arrow(
            pos=vector(0, 0, 0) + approximate_starting_location,
            length=EXTERNAL_REF_ARROW_LENGTH,
            shaftwidth=0.005,
            color=color.green,
            axis=vector(0, 0, 1),
            opacity=0.5,
        )

        # Arrows for tracker's local axes (X=Red, Y=Blue, Z=Green)
        x_axis = arrow(length=PROBE_ARROW_LENGTH, shaftwidth=0.01, color=color.red)
        y_axis = arrow(length=PROBE_ARROW_LENGTH, shaftwidth=0.01, color=color.blue)
        z_axis = arrow(length=PROBE_ARROW_LENGTH, shaftwidth=0.01, color=color.green)

        # Goal state arrow; should be the same color as the probe
        goal_state_arrow = arrow(
            shaftwidth=0.01,
            color=vector(0.5, 0.5, 1),
            visible=False,
            opacity=0.5,
        )

        # Box to represent the probe, which is in the y-axis direction of the tracker
        # Should be centered in the x and z directions
        probe_length = 0.09
        probe_width = 0.05
        probe_height = 0.03
        # Convert probe_center_offset to numpy array since Rotation.apply() expects array-like input
        probe_center_offset_array = np.array([0, -probe_length / 2, 0])
        probe_box = box(
            pos=vector(*probe_center_offset_array),
            length=probe_length,
            width=probe_width,
            height=probe_height,
            color=vector(0.5, 0.5, 1),
            opacity=0.5,
        )

        # State for goal visualization persistence
        active_goal_content = None
        goal_arrow_is_satisfied = False

        try:
            while True:
                # Fetch pose data from HTTP service
                pose_data, success = fetch_pose_from_server()
                latest_pose_data = pose_data  # Update for button callback access

                probe_position_current_frame = None  # Initialize for this frame

                if success and pose_data:
                    probe_position, pose_r_obj = format_pose(pose_data)

                    if probe_position is not None and pose_r_obj is not None:
                        probe_position_current_frame = (
                            probe_position  # Store valid position
                        )
                        current_time_epoch = pose_data.get("timestamp", time.time())
                        probe_quat_for_print = pose_r_obj.as_quat()

                        print(
                            f"time={current_time_epoch:.3f} "
                            f"pos  probe_position={probe_position}   "
                            f"rot_quat={probe_quat_for_print}"
                        )

                        # Update the position of the VPython axes
                        x_axis.pos = probe_position
                        y_axis.pos = probe_position
                        z_axis.pos = probe_position

                        # Rotate the probe's local axes by the transformed quaternion
                        x_axis.axis = vector(
                            *pose_r_obj.apply((-PROBE_ARROW_LENGTH, 0, 0))
                        )
                        y_axis.axis = vector(
                            *pose_r_obj.apply((0, -PROBE_ARROW_LENGTH, 0))
                        )
                        z_axis.axis = vector(
                            *pose_r_obj.apply((0, 0, PROBE_ARROW_LENGTH))
                        )

                        x_axis.visible = True
                        y_axis.visible = True
                        z_axis.visible = True

                        # Update the position of the probe box
                        print("Updating probe box position")
                        print(f"probe_position: {probe_position}")
                        print(f"probe_center_offset_array: {probe_center_offset_array}")
                        rotated_offset = pose_r_obj.apply(probe_center_offset_array)
                        # print(f"rotated_offset: {rotated_offset}")
                        probe_box.pos = probe_position + vector(*rotated_offset)

                        # Rotate the probe box by the transformed quaternion
                        probe_box.axis = vector(
                            *pose_r_obj.apply((0, -probe_length, 0))
                        )
                        probe_box.up = vector(*pose_r_obj.apply((0, 0, 1)))
                    else:
                        print(f"time={time.time():.3f} Invalid pose data received")
                        x_axis.visible = False
                        y_axis.visible = False
                        z_axis.visible = False
                        # probe_position_current_frame remains None
                else:
                    print(f"time={time.time():.3f} Tracker pose not available")
                    x_axis.visible = False
                    y_axis.visible = False
                    z_axis.visible = False
                    # probe_position_current_frame remains None

                # Update goal state visualization
                global latest_goal_state  # Read from the HTTP server thread

                if latest_goal_state is not None:
                    # Check if this is a new goal or if the active goal has changed
                    if latest_goal_state != active_goal_content:
                        active_goal_content = latest_goal_state
                        goal_arrow_is_satisfied = (
                            False  # New goal, so it's not satisfied yet
                        )

                    if not goal_arrow_is_satisfied:
                        try:
                            location = active_goal_content.get("location")
                            pose_vectors = active_goal_content.get("pose_vectors")

                            if location and pose_vectors and len(pose_vectors) > 0:
                                goal_pos_vpython = vector(
                                    location[0], location[1], location[2]
                                )
                                direction_vector = vector(
                                    pose_vectors[0],
                                    pose_vectors[1],
                                    pose_vectors[2],
                                )
                                goal_axis = direction_vector.norm() * GOAL_ARROW_LENGTH

                                goal_state_arrow.pos = goal_pos_vpython
                                goal_state_arrow.axis = goal_axis
                                goal_state_arrow.visible = True

                                # Proximity check if probe position is valid for this frame
                                if probe_position_current_frame is not None:
                                    # probe_box.pos is the center of the probe, updated if probe_position_current_frame is valid
                                    distance = (probe_box.pos - goal_pos_vpython).mag
                                    if distance < GOAL_PROXIMITY_THRESHOLD:
                                        goal_arrow_is_satisfied = True
                                        goal_state_arrow.visible = False
                            else:  # Invalid goal content structure
                                goal_state_arrow.visible = False
                        except Exception as e:
                            print(
                                f"Error processing active_goal_content for visualization: {e}"
                            )
                            goal_state_arrow.visible = False
                    else:  # Goal is currently satisfied
                        goal_state_arrow.visible = False
                else:  # latest_goal_state is None (no goal from server)
                    active_goal_content = None
                    goal_arrow_is_satisfied = False
                    goal_state_arrow.visible = False

                sys.stdout.flush()
                rate(24)  # Limit to ~24 Hz

        except Exception as e:  # Catches VPython window close or other loop errors
            import traceback

            print(f"Visualization loop terminated or error occurred:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            print(f"Line Number: {traceback.extract_tb(e.__traceback__)[-1].lineno}")
            print("Full traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("Visualization finished.")


if __name__ == "__main__":
    track_pose()
