# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sensor module for ultrasound data processing."""

import numpy as np
import quaternion as qt
from scipy.optimize import least_squares
from tbp.monty.frameworks.models.monty_base import SensorModuleBase
from tbp.monty.frameworks.models.states import State


class UltrasoundSM(SensorModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotting_data = {
            "column_points": [],
            "center_edge": None,
            "fitted_circle": None,
            "point_normal": None,
            "curvature": None,
            "mean_depth": 0.0,
            "observed_locations": [],
            "normal_rel_world": [],
        }

    def step(self, data):
        # Calculate patch pose in world coordinates
        tracker_position, probe_position, tracker_orientation, probe_orientation = (
            self.get_tracker_position_and_orientation(
                data["proprioceptive_state_agent"], data["proprioceptive_state_patch"]
            )
        )

        # normal_rel_patch will point up (in y direction). In the world frame it should
        # point towards the agent.
        normal_rel_patch, curvature, pixel_depth_in_patch = (
            self.extract_patch_pose_feat(data["img"])
        )

        # Derive depth from pixel location in image
        pixel_depth_in_image = data["patch_pixel_start"] + pixel_depth_in_patch
        data["patch_depth"] = self.get_depth_from_pixel_location(
            data["full_image_height"], pixel_depth_in_image
        )
        print(f"Depth in patch (in cm): {data['patch_depth'] * 100}")

        patch_world_location = self.get_patch_world_location(
            tracker_position,
            probe_position,
            tracker_orientation,
            data["patch_depth"],
        )

        self.plotting_data["observed_locations"].append(patch_world_location)

        # apply tracker orientation quaternion to patch pose feat
        # To ensure normal_rel_world points towards the agent, we negate normal_rel_patch.
        # TODO: make sure first curvature direction is aligned with plane of image
        normal_rel_world = qt.as_rotation_matrix(tracker_orientation) @ (
            -normal_rel_patch
        )
        self.plotting_data["normal_rel_world"].append(normal_rel_world)
        # Add two orthonormal vectors to form a basis for the patch's orientation in the world frame.
        patch_world_orientation = self.calculate_patch_world_orientation(
            normal_rel_world
        )

        # Store data for plotting
        self.plotting_data["mean_depth"] = data["patch_depth"]
        self.plotting_data["point_normal"] = normal_rel_patch
        self.plotting_data["curvature"] = curvature

        CMP_output = State(
            location=patch_world_location,
            morphological_features={
                "pose_vectors": patch_world_orientation,
                "pose_fully_defined": False,
            },
            non_morphological_features={
                # NOTE: This will not match with any of the curvature features in the
                # pretrained models from the simulation.
                "curvature": curvature,
            },
            confidence=1,
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )
        return CMP_output

    def get_depth_from_pixel_location(
        self, full_image_height, pixel_depth_in_image, max_depth=7
    ):
        """Crops image and then calculates percentage from top of pixel location.

        Args:
            full_image_height: The height of the full ultrasound image in pixels
            pixel_depth_in_image: The y pixel coordinate of the depth reading within the image
            max_depth: Depth setting of the ultrasound probe, in cm

        Returns:
            float: Estimated depth of patch in meters
        """
        depth_perc = pixel_depth_in_image / full_image_height
        depth_cm = depth_perc * max_depth
        depth_m = depth_cm / 100

        return depth_m

    def get_patch_world_location(
        self,
        agent_position,
        sensor_position,
        agent_rotation,
        image_depth,
    ):
        """Calculates the patch's location and orientation in world coordinates."""
        # adding the 0.03
        offset_direction = np.array([0.0, 0.0, 1.0])
        rotated_offset_direction = (
            qt.as_rotation_matrix(agent_rotation) @ offset_direction
        )
        offset_distance = sensor_position[1]  # = 0.03
        relative_offset = offset_distance * rotated_offset_direction
        agent_position = agent_position + relative_offset

        # adding the 0.095 + depth
        fake_sensor_rel_world = qt.as_rotation_matrix(agent_rotation) @ np.array(
            [0.0, 1.0, 0.0]
        )
        offset_distance = sensor_position[2] + image_depth

        # Determine the movement vector.
        # The vector is in the opposite direction of the unit_normal (-unit_normal)
        # and scaled by the offset_distance.
        movement_vector = -fake_sensor_rel_world * offset_distance

        # Compute the new location by applying the movement_vector to the agent's current position.
        patch_world_location = agent_position + movement_vector

        return patch_world_location

    def calculate_patch_world_orientation(self, patch_normal_in_world_frame):
        """Calculates the patch's orientation in the world frame.

        Args:
            patch_normal_in_world_frame (np.ndarray): The patch's surface normal vector,
                                                      already expressed in world coordinates.

        Returns:
            np.ndarray: A 3x3 matrix representing the patch's orientation in the world.
                        The rows of the matrix are orthonormal basis vectors (normal, dir1, dir2)
                        expressed in world coordinates.
        """
        # make sure patch_normal_in_world_frame is normalized.
        normal_vector = patch_normal_in_world_frame / np.linalg.norm(
            patch_normal_in_world_frame
        )

        # Create a coordinate system where normal_vector is the first basis vector.
        # dir1 will be orthogonal to normal_vector.
        # dir2 will be orthogonal to both normal_vector and dir1, forming a right-handed system.

        # To find dir1, we can take the cross product with a fixed world axis (e.g., Z-axis).
        # This ensures dir1 is generally in the world XY plane or derived consistently.
        world_z_axis = np.array([0.0, 0.0, 1.0])
        dir1 = np.cross(world_z_axis, normal_vector)

        # Handle the case where normal_vector is parallel to world_z_axis
        if np.linalg.norm(dir1) < 1e-6:
            # If normal_vector is along Z, dir1 can be set to world X-axis.
            dir1 = np.array([1.0, 0.0, 0.0])

        dir1 /= np.linalg.norm(dir1)  # Normalize dir1

        # Calculate dir2 as the cross product of normal_vector and dir1
        # This ensures (normal_vector, dir1, dir2) forms a right-handed orthonormal basis.
        dir2 = np.cross(normal_vector, dir1)
        # dir2 should already be normalized if normal_vector and dir1 are orthonormal.
        # Adding normalization for robustness.
        dir2 /= np.linalg.norm(dir2)

        # Construct the patch orientation matrix with these basis vectors as rows.
        # This matrix represents the orientation of the patch in the world frame.
        patch_orientation_in_world = np.vstack([normal_vector, dir1, dir2])

        # The matrix 'patch_orientation_in_world' is already the desired orientation.
        # The previous multiplication by tracker_orientation_matrix was incorrect here,
        # as the basis vectors are already in the world frame.
        return patch_orientation_in_world

    def get_tracker_position_and_orientation(
        self, proprioceptive_state_agent, proprioceptive_state_probe
    ):
        # TODO: make sure we don't need to account for sensor rel. agent position in our
        # scenario (input proprioceptive_state_patch additionally)
        tracker_position = np.array(proprioceptive_state_agent["position"])
        probe_position = np.array(proprioceptive_state_probe["position"])
        # convert if it is not a quaternion
        if isinstance(proprioceptive_state_agent["rotation"], qt.quaternion):
            tracker_orientation = proprioceptive_state_agent["rotation"]
            probe_orientation = proprioceptive_state_probe["rotation"]
        else:
            # convert to quaternion from list
            tracker_orientation = qt.quaternion(
                proprioceptive_state_agent["rotation"][0],
                proprioceptive_state_agent["rotation"][1],
                proprioceptive_state_agent["rotation"][2],
                proprioceptive_state_agent["rotation"][3],
            )
            probe_orientation = qt.quaternion(
                proprioceptive_state_probe["rotation"][0],
                proprioceptive_state_probe["rotation"][1],
                proprioceptive_state_probe["rotation"][2],
                proprioceptive_state_probe["rotation"][3],
            )
        return tracker_position, probe_position, tracker_orientation, probe_orientation

    def update_state(self, state):
        """Currently pass state info to step function.

        TODO: should probably better do this here.
        """
        pass

    def extract_patch_pose_feat(self, patch):
        """Extract patch pose features including curvature and surface normal from a 256x256 grayscale image.

        First extracts edge points, then fits a circle to these points, and finally calculates
        the normal vector based on the fitted curve at the center location.

        Args:
            patch (np.ndarray): 256x256 grayscale image containing a white edge on black background

        Returns:
            tuple: (normal_3d, curvature) where normal_3d is a 3D numpy array
                  representing the surface normal in the image plane (e.g., [nx, ny, 0]),
                  and curvature is a float representing the local curvature.
        """
        # Get points on the first edge (from top) in the image
        all_column_points_tuples, center_edge_point = self.extract_edge_points(patch)
        all_column_points_np = np.array(all_column_points_tuples)

        # Store data for plotting
        self.plotting_data["column_points"] = all_column_points_tuples
        self.plotting_data["center_edge"] = center_edge_point

        # Fit a circle to the edge points and calculate curvature
        fit_params, curvature = self.fit_circle_to_points(
            patch, all_column_points_np, center_edge_point
        )

        # Store circle fit parameters for plotting if available
        if fit_params is not None and not fit_params.get("is_line", True):
            self.plotting_data["fitted_circle"] = (
                fit_params["center"][0],
                fit_params["center"][1],
                fit_params["radius"],
            )
        else:
            self.plotting_data["fitted_circle"] = None

        # Calculate the normal vector based on the fitted curve
        normal_3d = self.calculate_normal_from_fit(fit_params, center_edge_point)

        y_depth_in_image = center_edge_point[1]

        return normal_3d, curvature, y_depth_in_image

    def extract_edge_points(self, image):
        """Extract edge points from the image by scanning each column for intensity values
        above a threshold.

        Args:
            image (np.ndarray): Grayscale image (e.g., 256x256).

        Returns:
            tuple: (all_column_points_tuples, center_edge_point) where:
                - all_column_points_tuples is a list of (x,y) tuples representing edge points
                - center_edge_point is the (x,y) coordinates at the center column
        """
        image_height, image_width = image.shape[:2]

        # Calculate adaptive white threshold based on image max intensity
        max_intensity = np.max(image)
        WHITE_THRESHOLD = max(50, 0.3 * max_intensity)

        # range around center y to scan for edge points
        Y_SCAN_RANGE = 50
        center_x = image_width // 2
        y_center = image_height // 2  # Default center y if no edge is found

        # Find approximate center y by looking at center column
        for y_row in range(image_height):
            if image[y_row, center_x] > WHITE_THRESHOLD:
                y_center = y_row
                break

        y_min_scan = max(0, y_center - Y_SCAN_RANGE)
        y_max_scan = min(image_height, y_center + Y_SCAN_RANGE + 1)

        all_column_points_tuples = []  # List of (x,y) tuples
        center_edge_point = None

        for x_col in range(image_width):
            for y_row in range(y_min_scan, y_max_scan):
                if image[y_row, x_col] > WHITE_THRESHOLD:
                    point = (float(x_col), float(y_row))
                    all_column_points_tuples.append(point)

                    # Save the center edge point
                    if x_col == center_x:
                        center_edge_point = point
                    break

        # If no center point was found, use the middle of the image
        if center_edge_point is None:
            center_edge_point = (float(center_x), float(y_center))

        return all_column_points_tuples, center_edge_point

    def fit_circle_to_points(self, image, all_column_points_np, edge_point):
        """Fits a circle to the detected edge points to get curvature.

        Also makes sure the circle passes through the edge point at the center column.

        Args:
            image (np.ndarray): The input image for visualization.
            all_column_points_np (np.ndarray): Array of (x,y) points representing the edge.
            edge_point (tuple): (x,y) coordinates of the center edge point.

        Returns:
            tuple: (fit_params, curvature) where:
                - fit_params is a dictionary containing circle fit parameters
                - curvature is the calculated curvature value
        """
        # Ensure all_column_points_np is a numpy array
        if not isinstance(all_column_points_np, np.ndarray):
            all_column_points_np = np.array(all_column_points_np)

        # Check if we have enough points to fit a circle (need at least 3)
        if len(all_column_points_np) < 3:
            # Not enough points for a meaningful fit
            fit_params = {
                "is_line": True,
                "points_used": [edge_point],
                "retry_count": 0,
            }
            final_curvature = 0.0
            return fit_params, final_curvature

        # Define function to fit a circle through P0 and minimize geometric error
        def fit_circle_through_point(points, P0, calc_error=False):
            """
            Fit a circle (a,b,r) to `points` minimising geometric
            error, while forcing the circle to pass exactly through P0.

            Parameters
            ----------
            points : (N,2) array-like
                Points to fit circle to
            P0 : length-2 iterable
                The fixed point (x0,y0) the circle must pass through
            calc_error : bool, optional
                If True, return the mean squared residual as a measure of fit quality

            Returns
            -------
            tuple
                ((a,b), r, error) if calc_error=True, else ((a,b), r)
                where (a,b) is the center, r is the radius, and error is the MSE
            """
            pts = np.asarray(points, dtype=float)
            x0, y0 = P0

            # Helper: residuals for LSQ over (a,b) only
            def residuals(ab):
                a, b = ab
                r = np.hypot(x0 - a, y0 - b)  # constraint - circle passes through P0
                return np.hypot(pts[:, 0] - a, pts[:, 1] - b) - r

            # Initial guess using crude algebraic fit
            x, y = pts[:, 0], pts[:, 1]
            A = np.c_[2 * x, 2 * y, np.ones_like(x)]
            try:
                c, _, _, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
                a0, b0 = c[0], c[1]
            except np.linalg.LinAlgError:
                # Fallback if algebraic fit fails
                a0, b0 = np.mean(pts[:, 0]), np.mean(pts[:, 1])

            try:
                res = least_squares(residuals, x0=[a0, b0], method="trf")
                a, b = res.x
                r = np.hypot(x0 - a, y0 - b)  # exact by construction

                if calc_error:
                    # Calculate mean squared error
                    final_residuals = residuals([a, b])
                    mse = np.mean(np.square(final_residuals))
                    return (a, b), r, mse
                else:
                    return (a, b), r
            except Exception:
                # In case optimization fails
                if calc_error:
                    return None, float("inf"), float("inf")
                else:
                    return None, float("inf")

        # Function to normalize points for numerical stability
        def normalize_and_fit(points, P0):
            # Normalize data for numerical stability
            mean_x = np.mean(points[:, 0])
            mean_y = np.mean(points[:, 1])
            std_xy = np.std(points, axis=0).mean()

            # Guard against zero standard deviation
            if std_xy < 1e-8:
                std_xy = 1.0

            # Normalize points and the fixed point P0
            norm_points = (points - [mean_x, mean_y]) / std_xy
            norm_P0 = ((P0[0] - mean_x) / std_xy, (P0[1] - mean_y) / std_xy)

            # Fit circle to normalized points (with error calculation)
            center_norm, radius_norm, fit_error = fit_circle_through_point(
                norm_points, norm_P0, calc_error=True
            )

            return center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy

        # First attempt with all points
        center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy = normalize_and_fit(
            all_column_points_np, edge_point
        )

        # Decide whether to use all points or a subset
        use_subset = False
        best_subset_error = float("inf")
        best_subset_params = None
        best_subset_points = None
        attempt_num_for_best_subset = 0
        retry_count = 0

        # TODO: make these function parameters
        subset_percentages = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        min_fit_error = 0.05

        for percentage in subset_percentages:
            retry_count += 1

            # Calculate how many points to keep from each end
            total_points = len(all_column_points_np)
            points_to_keep = int(total_points * percentage)

            # Ensure we keep a minimum number of points
            if points_to_keep < 10:
                points_to_keep = min(10, total_points)

            # Determine start/end indices for subsetting
            points_to_remove = (total_points - points_to_keep) // 2
            if 2 * points_to_remove + points_to_keep > total_points:
                points_to_remove = (total_points - points_to_keep) // 2

            start_idx = points_to_remove
            end_idx = total_points - points_to_remove

            # Safety check
            if start_idx < 0:
                start_idx = 0
            if end_idx > total_points:
                end_idx = total_points
            if end_idx - start_idx < 3:
                print(
                    f"  Retry {retry_count}: Not enough points to subset ({end_idx - start_idx})"
                )
                continue

            # Create subset by removing points from start and end
            subset_points = all_column_points_np[start_idx:end_idx]

            # Try fit with subset
            try:
                (
                    center_norm_subset,
                    radius_norm_subset,
                    fit_error_subset,
                    mean_x_subset,
                    mean_y_subset,
                    std_xy_subset,
                ) = normalize_and_fit(subset_points, edge_point)

                # Sanity check on the radius
                if radius_norm_subset <= 0 or radius_norm_subset > 1e6:
                    print(f"    Skipping: Invalid radius {radius_norm_subset}")
                    continue

                # Update best subset if this is better
                if fit_error_subset < best_subset_error:
                    best_subset_error = fit_error_subset
                    best_subset_params = (
                        center_norm_subset,
                        radius_norm_subset,
                        fit_error_subset,
                        mean_x_subset,
                        mean_y_subset,
                        std_xy_subset,
                    )
                    best_subset_points = subset_points
                    attempt_num_for_best_subset = retry_count

            except Exception as e:
                print(f"  Error in retry {retry_count}: {str(e)}")

        # Decide whether to use the subset
        if best_subset_params is not None:
            # Always use subset if original has high error or bad radius
            if fit_error > min_fit_error or radius_norm <= 0 or radius_norm > 1e6:
                use_subset = True

            # If using subset, update parameters
            if use_subset:
                center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy = (
                    best_subset_params
                )
                points_to_use = best_subset_points
            else:
                points_to_use = all_column_points_np
        else:
            # No valid subset found, use original
            points_to_use = all_column_points_np

        final_displayed_retry_count = 0
        if use_subset:
            final_displayed_retry_count = attempt_num_for_best_subset

        # Check if the fit is valid
        if center_norm is None or radius_norm > 1e6 or radius_norm < 1e-6:
            # The points are nearly collinear or the fit is unstable
            fit_params = {
                "is_line": True,
                "points_used": [edge_point],
                "retry_count": final_displayed_retry_count,
            }
            final_curvature = 0.0
        else:
            # Denormalize the results
            center_x = center_norm[0] * std_xy + mean_x
            center_y = center_norm[1] * std_xy + mean_y
            radius = radius_norm * std_xy

            # Calculate curvature (1/radius)
            final_curvature = 1.0 / radius if radius > 1e-6 else 0.0

            # Convert points to tuples for visualization
            points_tuples = [tuple(pt) for pt in points_to_use]

            # Store the fit parameters for visualization
            fit_params = {
                "is_line": False,
                "center": (center_x, center_y),
                "radius": radius,
                "points_used": [edge_point]
                + points_tuples,  # Include P0 and used points
                "inliers_percent": len(points_to_use) / len(all_column_points_np)
                if len(all_column_points_np) > 0
                else 0.0,
                "retry_count": final_displayed_retry_count,
                "fit_error": fit_error,
                "used_subset": use_subset,
            }

        return fit_params, final_curvature

    def calculate_normal_from_fit(self, fit_params, edge_point):
        """Calculate the normal vector based on the fitted circle/line.

        Args:
            fit_params (dict): Dictionary containing fit parameters.
            edge_point (tuple): (x,y) of the point where we want to calculate the normal.

        Returns:
            np.ndarray: 3D normal vector [nx, ny, 0]
        """
        x0, y0 = edge_point

        if fit_params.get("is_line", True):
            # For a line (or failed fit), default to pointing upward
            normal = np.array([0.0, -1.0, 0.0])
        else:
            # For a circle, the normal points from the center to the edge point
            circle_center = fit_params["center"]
            cx, cy = circle_center

            # Vector from center to edge point
            dx = x0 - cx
            dy = y0 - cy

            # Normalize
            magnitude = np.hypot(dx, dy)
            if magnitude > 1e-9:
                nx = dx / magnitude
                ny = dy / magnitude
            else:
                # Default if points are very close
                nx, ny = 0.0, -1.0

            # Ensure normal always points upward (negative y in image coordinates)
            if ny > 0:
                nx = -nx
                ny = -ny

            normal = np.array([nx, ny, 0.0])

        return normal
