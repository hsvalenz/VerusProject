# import os
# 
# import gym
# import numpy as np
# from gym.utils import seeding
# 
# import matplotlib
# 
# # Use non-interactive backend so this works without a GUI.
# matplotlib.use("Agg")
# 
# import matplotlib.pyplot as plt
# 
# 
# class SimpleInspectionEnv(gym.Env):
#     """
#     Simplified single-inspector environment.
# 
#     Observation:
#         [
#             x, y, z,
#             vx, vy, vz,
#             sun_angle,
#             num_inspected,
#             target_dir_x, target_dir_y, target_dir_z
#         ]
# 
#     Action:
#         Discrete action mapped to 3D thrust [x, y, z].
# 
#     Task:
#         Inspect discrete surface points on the RSO/chief.
# 
#     Camera assumption:
#         The deputy camera always points toward the RSO/chief center.
#         A point is inspected if:
#             1. deputy is in the viewing shell
#             2. point is inside the camera cone
#             3. point is on the visible/front-facing hemisphere
# 
#     Exploration:
#         The environment rewards angular motion around the RSO and gives
#         a one-time bonus for reaching the opposite side of the sphere.
#     """
# 
#     def __init__(self, args=None):
#         super().__init__()
# 
#         self.args = args
# 
#         # Rendering is OFF by default.
#         self.render_enabled = False
#         self.frame_dir = "frames"
#         self.render_frame_idx = 0
# 
#         # Observation:
#         # [x, y, z, vx, vy, vz, sun_angle, num_inspected, tx, ty, tz]
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(11,),
#             dtype=np.float32,
#         )
# 
#         # 3 bins per axis -> 3^3 = 27 actions
#         self.num_bins = 3
#         self.action_space = gym.spaces.Discrete(self.num_bins ** 3)
# 
#         # Dynamics parameters
#         self.dt = 1.0
#         self.thrust_scale = 0.35
#         self.velocity_damping = 0.96
#         self.max_time = 700
# 
#         # RSO / chief position
#         self.target_pos = np.zeros(3, dtype=np.float32)
# 
#         # Safety geometry
#         self.crash_radius = 8.0
#         self.max_distance = 220.0
# 
#         # Inspection/viewing shell
#         self.success_inner_radius = 15.0
#         self.success_outer_radius = 25.0
# 
#         # Put inspection points exactly on the inner sphere.
#         self.inspection_point_radius = self.success_inner_radius
# 
#         # Deputy should view those inner-sphere points from farther out.
#         self.desired_distance = 20.0
# 
#         # Discrete inspection points on the RSO inner sphere.
#         self.num_inspection_points = 100
#         self.success_threshold = self.num_inspection_points
# 
#         # Camera model:
#         # The camera always points toward the RSO center.
#         # A point is inspected if it lies inside this half-angle cone.
#         self.camera_half_angle = np.deg2rad(7.5)
# 
#         # Only allow inspection while deputy is in this viewing shell.
#         self.view_inner_radius = self.success_inner_radius
#         self.view_outer_radius = 35.0
# 
#         # Reward magnitudes
#         self.new_point_reward = 10.0
#         self.all_points_success_reward = 200.0
#         self.crash_penalty = -250.0
#         self.out_of_bounds_penalty = -100.0
# 
#         # Dense shaping scales
#         self.viewpoint_progress_scale = 1.5
#         self.radius_progress_scale = 0.5
#         self.view_shell_reward = 0.25
# 
#         # Exploration reward terms
#         self.angular_motion_reward_scale = 2.0
#         self.new_visible_patch_reward = 0.5
# 
#         # Opposite-side traversal reward
#         self.opposite_side_reward = 25.0
#         self.opposite_side_dot_threshold = -0.75
# 
#         # Target selection pressure.
#         # Higher means target selection prefers farther/opposite unvisited points.
#         self.opposite_target_reward_scale = 2.5
# 
#         # Penalties
#         self.thrust_penalty_scale = 0.02
#         self.speed_penalty_scale = 0.01
#         self.unsafe_zone_penalty_scale = 1.0
#         self.no_new_point_penalty = -0.01
# 
#         # Penalize staying at the same angular location without inspecting.
#         self.stuck_penalty = -0.25
#         self.stuck_angle_threshold = np.deg2rad(1.0)
# 
#         # Penalize staying on the starting hemisphere too long.
#         self.same_hemisphere_penalty = -0.05
# 
#         self.seed()
#         self.reset()
# 
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
# 
#     def enable_render(self, frame_dir="frames", clear_old_frames=False):
#         """
#         Enable rendering for a single rollout.
# 
#         This should be called only after training finishes, from train/oracle.py.
#         """
#         self.render_enabled = True
#         self.frame_dir = frame_dir
#         self.render_frame_idx = 0
# 
#         if clear_old_frames and os.path.exists(self.frame_dir):
#             import shutil
#             shutil.rmtree(self.frame_dir)
# 
#         os.makedirs(self.frame_dir, exist_ok=True)
# 
#         print(f"[SimpleInspectionEnv] Rendering enabled. Saving frames to: {self.frame_dir}/")
# 
#     def disable_render(self):
#         self.render_enabled = False
# 
#     def render(self, mode="human"):
#         os.makedirs(self.frame_dir, exist_ok=True)
# 
#         fig = plt.figure(figsize=(8, 7))
#         ax = fig.add_subplot(111, projection="3d")
# 
#         # RSO / chief
#         ax.scatter(
#             self.target_pos[0],
#             self.target_pos[1],
#             self.target_pos[2],
#             s=140,
#             label="RSO / Chief",
#         )
# 
#         # Deputy
#         ax.scatter(
#             self.pos[0],
#             self.pos[1],
#             self.pos[2],
#             s=90,
#             label="Deputy",
#         )
# 
#         # Inspection points
#         visited_points = self.inspection_points[self.inspected_mask]
#         unvisited_points = self.inspection_points[~self.inspected_mask]
# 
#         if len(unvisited_points) > 0:
#             ax.scatter(
#                 unvisited_points[:, 0],
#                 unvisited_points[:, 1],
#                 unvisited_points[:, 2],
#                 s=12,
#                 alpha=0.45,
#                 label="Uninspected surface points",
#             )
# 
#         if len(visited_points) > 0:
#             ax.scatter(
#                 visited_points[:, 0],
#                 visited_points[:, 1],
#                 visited_points[:, 2],
#                 s=20,
#                 alpha=0.95,
#                 label="Inspected surface points",
#             )
# 
#         # Current target surface point and desired viewing position
#         target_point = self._get_current_target_point()
#         target_viewpoint = self._get_current_target_viewpoint()
# 
#         if target_point is not None:
#             ax.scatter(
#                 target_point[0],
#                 target_point[1],
#                 target_point[2],
#                 s=110,
#                 marker="x",
#                 label="Current surface target",
#             )
# 
#         if target_viewpoint is not None:
#             ax.scatter(
#                 target_viewpoint[0],
#                 target_viewpoint[1],
#                 target_viewpoint[2],
#                 s=90,
#                 marker="^",
#                 label="Desired viewing position",
#             )
# 
#             ax.plot(
#                 [self.pos[0], target_viewpoint[0]],
#                 [self.pos[1], target_viewpoint[1]],
#                 [self.pos[2], target_viewpoint[2]],
#                 linestyle="--",
#                 linewidth=1.0,
#                 label="Path to viewing position",
#             )
# 
#         # Trajectory
#         if len(self.trajectory) > 1:
#             traj = np.asarray(self.trajectory)
#             ax.plot(
#                 traj[:, 0],
#                 traj[:, 1],
#                 traj[:, 2],
#                 linewidth=1.5,
#                 label="Trajectory",
#             )
# 
#         # Velocity vector
#         ax.quiver(
#             self.pos[0],
#             self.pos[1],
#             self.pos[2],
#             self.vel[0],
#             self.vel[1],
#             self.vel[2],
#             length=4.0,
#             normalize=False,
#             label="Velocity",
#         )
# 
#         # Camera pointing direction: always toward RSO
#         camera_forward = self._safe_unit(self.target_pos - self.pos)
#         ax.quiver(
#             self.pos[0],
#             self.pos[1],
#             self.pos[2],
#             camera_forward[0],
#             camera_forward[1],
#             camera_forward[2],
#             length=10.0,
#             normalize=False,
#             label="Camera forward",
#         )
# 
#         # Radius spheres
#         self._draw_sphere(ax, self.crash_radius, alpha=0.30)
#         self._draw_sphere(ax, self.success_inner_radius, alpha=0.15)
#         self._draw_sphere(ax, self.success_outer_radius, alpha=0.15)
# 
#         limit = 75
#         ax.set_xlim(-limit, limit)
#         ax.set_ylim(-limit, limit)
#         ax.set_zlim(-limit, limit)
# 
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")
# 
#         distance = np.linalg.norm(self.pos - self.target_pos)
#         num_inspected = int(np.sum(self.inspected_mask))
#         start_side_dot = float(np.dot(self._get_current_view_dir(), self.start_view_dir))
# 
#         ax.set_title(
#             f"frame={self.render_frame_idx}, "
#             f"t={self.t}, "
#             f"distance={distance:.2f}, "
#             f"inspected={num_inspected}/{self.num_inspection_points}, "
#             f"opposite={self.reached_opposite_side}, "
#             f"start_dot={start_side_dot:.2f}, "
#             f"status={self.status}"
#         )
# 
#         ax.legend(loc="upper right", fontsize=8)
# 
#         filename = os.path.join(
#             self.frame_dir,
#             f"frame_{self.render_frame_idx:04d}.png",
#         )
# 
#         plt.savefig(filename)
#         plt.close(fig)
# 
#         self.render_frame_idx += 1
# 
#     def _draw_sphere(self, ax, radius, alpha=0.25):
#         u = np.linspace(0, 2 * np.pi, 24)
#         v = np.linspace(0, np.pi, 12)
# 
#         x = radius * np.outer(np.cos(u), np.sin(v))
#         y = radius * np.outer(np.sin(u), np.sin(v))
#         z = radius * np.outer(np.ones_like(u), np.cos(v))
# 
#         ax.plot_wireframe(
#             x + self.target_pos[0],
#             y + self.target_pos[1],
#             z + self.target_pos[2],
#             alpha=alpha,
#         )
# 
#     def _generate_sphere_directions(self, n):
#         """
#         Generate approximately evenly spaced unit directions using the
#         Fibonacci sphere method.
#         """
#         directions = []
#         golden_angle = np.pi * (3.0 - np.sqrt(5.0))
# 
#         for i in range(n):
#             if n == 1:
#                 y = 0.0
#             else:
#                 y = 1.0 - (2.0 * i) / float(n - 1)
# 
#             radius_xy = np.sqrt(max(0.0, 1.0 - y * y))
#             theta = golden_angle * i
# 
#             x = np.cos(theta) * radius_xy
#             z = np.sin(theta) * radius_xy
# 
#             directions.append(np.array([x, y, z], dtype=np.float32))
# 
#         return np.asarray(directions, dtype=np.float32)
# 
#     def _generate_inspection_points(self):
#         """
#         Generate inspection points on the inner sphere.
#         """
#         directions = self._generate_sphere_directions(self.num_inspection_points)
#         points = self.target_pos + self.inspection_point_radius * directions
#         return points.astype(np.float32)
# 
#     def _map_action(self, action):
#         """
#         Converts one discrete action into [x_thrust, y_thrust, z_thrust].
# 
#         With num_bins = 3:
#             values = [-1, 0, 1]
#         """
#         raw = np.array(
#             np.unravel_index(
#                 int(action),
#                 (self.num_bins, self.num_bins, self.num_bins),
#             )
#         )
# 
#         values = np.linspace(-1.0, 1.0, self.num_bins, dtype=np.float32)
#         return values[raw]
# 
#     def _safe_unit(self, vec, eps=1e-8):
#         vec = np.asarray(vec, dtype=np.float32)
#         norm = np.linalg.norm(vec)
# 
#         if norm < eps:
#             return np.zeros_like(vec, dtype=np.float32)
# 
#         return (vec / norm).astype(np.float32)
# 
#     def _angle_between(self, a, b):
#         a = self._safe_unit(a)
#         b = self._safe_unit(b)
# 
#         dot = np.clip(np.dot(a, b), -1.0, 1.0)
#         return float(np.arccos(dot))
# 
#     def _get_unvisited_indices(self):
#         return np.where(~self.inspected_mask)[0]
# 
#     def _get_current_view_dir(self):
#         """
#         Direction from RSO center to deputy.
#         """
#         return self._safe_unit(self.pos - self.target_pos)
# 
#     def _get_current_target_index(self):
#         """
#         Pick an unvisited inspection point that encourages traversal.
# 
#         Simpler than sector logic:
#             - prefer unvisited points whose desired viewpoint is far from
#               the current deputy viewpoint
#             - this encourages going around the RSO without maintaining
#               explicit view sectors
#         """
#         unvisited = self._get_unvisited_indices()
# 
#         if len(unvisited) == 0:
#             return None
# 
#         current_view_dir = self._get_current_view_dir()
# 
#         best_idx = None
#         best_score = -np.inf
# 
#         for idx in unvisited:
#             point = self.inspection_points[idx]
# 
#             desired_view_dir = self._safe_unit(point - self.target_pos)
# 
#             angle_from_current = self._angle_between(current_view_dir, desired_view_dir)
# 
#             # dot_current = 1: same side
#             # dot_current = -1: opposite side
#             dot_current = float(np.dot(current_view_dir, desired_view_dir))
#             opposite_score = -dot_current
# 
#             score = 0.0
#             score += self.opposite_target_reward_scale * angle_from_current
#             score += 2.0 * opposite_score
# 
#             if score > best_score:
#                 best_score = score
#                 best_idx = int(idx)
# 
#         return best_idx
# 
#     def _get_current_target_point(self):
#         idx = self._get_current_target_index()
# 
#         if idx is None:
#             return None
# 
#         return self.inspection_points[idx]
# 
#     def _get_current_target_viewpoint(self):
#         """
#         Return a desired deputy viewpoint for the current unvisited inspection point.
# 
#         The inspection point lies on the inner sphere. The desired viewpoint is
#         farther outward along the same radial direction.
#         """
#         target_point = self._get_current_target_point()
# 
#         if target_point is None:
#             return None
# 
#         surface_dir = self._safe_unit(target_point - self.target_pos)
#         viewpoint = self.target_pos + surface_dir * self.desired_distance
# 
#         return viewpoint.astype(np.float32)
# 
#     def _get_visible_unvisited_indices(self):
#         """
#         Return unvisited inspection point indices currently inside the camera cone
#         and on the visible/front-facing hemisphere.
# 
#         This does not modify inspected_mask.
#         """
#         unvisited = self._get_unvisited_indices()
# 
#         if len(unvisited) == 0:
#             return np.array([], dtype=int)
# 
#         distance_to_rso = np.linalg.norm(self.pos - self.target_pos)
# 
#         in_viewing_shell = (
#             self.view_inner_radius <= distance_to_rso <= self.view_outer_radius
#         )
# 
#         if not in_viewing_shell:
#             return np.array([], dtype=int)
# 
#         # Camera points from deputy toward RSO center.
#         camera_forward = self._safe_unit(self.target_pos - self.pos)
# 
#         unvisited_points = self.inspection_points[unvisited]
# 
#         # Direction from deputy to each surface point.
#         vectors_to_points = unvisited_points - self.pos
#         distances_to_points = np.linalg.norm(vectors_to_points, axis=1)
# 
#         valid = distances_to_points > 1e-8
# 
#         if not np.any(valid):
#             return np.array([], dtype=int)
# 
#         point_dirs_from_deputy = np.zeros_like(vectors_to_points, dtype=np.float32)
#         point_dirs_from_deputy[valid] = (
#             vectors_to_points[valid] / distances_to_points[valid, None]
#         )
# 
#         # Cone test:
#         # Is the surface point inside the camera cone?
#         cos_angles = point_dirs_from_deputy @ camera_forward
#         cos_threshold = np.cos(self.camera_half_angle)
#         inside_camera_cone = cos_angles >= cos_threshold
# 
#         # Front-side / occlusion test:
#         # Surface normal points outward from RSO center to the surface point.
#         surface_normals = np.zeros_like(unvisited_points, dtype=np.float32)
# 
#         for i, point in enumerate(unvisited_points):
#             surface_normals[i] = self._safe_unit(point - self.target_pos)
# 
#         # Direction from surface point to deputy.
#         point_to_deputy = self.pos - unvisited_points
#         point_to_deputy_distances = np.linalg.norm(point_to_deputy, axis=1)
# 
#         point_to_deputy_dirs = np.zeros_like(point_to_deputy, dtype=np.float32)
#         valid_los = point_to_deputy_distances > 1e-8
# 
#         point_to_deputy_dirs[valid_los] = (
#             point_to_deputy[valid_los] / point_to_deputy_distances[valid_los, None]
#         )
# 
#         # If dot > 0, the surface normal points toward the deputy.
#         # If dot < 0, the point is on the far side of the RSO and is occluded.
#         front_facing = np.sum(surface_normals * point_to_deputy_dirs, axis=1) > 0.0
# 
#         visible_local = np.where(inside_camera_cone & front_facing)[0]
# 
#         if len(visible_local) == 0:
#             return np.array([], dtype=int)
# 
#         return unvisited[visible_local]
# 
#     def _mark_inspected_points(self):
#         """
#         Mark currently visible unvisited points as inspected.
#         """
#         visible_indices = self._get_visible_unvisited_indices()
# 
#         if len(visible_indices) == 0:
#             return 0
# 
#         self.inspected_mask[visible_indices] = True
#         return int(len(visible_indices))
# 
#     def reset(self):
#         """
#         Old Gym API reset.
# 
#         Stable-Baselines3 with older Gym expects:
#             obs = env.reset()
#         """
#         self.pos = np.array([50.0, 0.0, 0.0], dtype=np.float32)
#         self.vel = np.zeros(3, dtype=np.float32)
# 
#         self.sun_angle = 0.0
#         self.t = 0
#         self.status = "Running"
# 
#         self.inspection_points = self._generate_inspection_points()
#         self.inspected_mask = np.zeros(self.num_inspection_points, dtype=bool)
# 
#         self.prev_visible_signature = None
#         self.trajectory = [self.pos.copy()]
# 
#         self.start_view_dir = self._get_current_view_dir()
#         self.reached_opposite_side = False
# 
#         return self._get_obs()
# 
#     def step(self, action):
#         assert self.action_space.contains(action), f"Invalid action: {action}"
# 
#         thrust_cmd = self._map_action(action)
# 
#         prev_distance_to_rso = np.linalg.norm(self.pos - self.target_pos)
#         prev_radius_error = abs(prev_distance_to_rso - self.desired_distance)
# 
#         prev_target_viewpoint = self._get_current_target_viewpoint()
# 
#         if prev_target_viewpoint is not None:
#             prev_target_distance = np.linalg.norm(prev_target_viewpoint - self.pos)
#         else:
#             prev_target_distance = 0.0
# 
#         prev_view_dir = self._get_current_view_dir()
# 
#         # Simple damped dynamics.
#         self.vel = (
#             self.velocity_damping * self.vel
#             + self.thrust_scale * thrust_cmd * self.dt
#         )
#         self.pos = self.pos + self.vel * self.dt
# 
#         self.t += 1
#         self.trajectory.append(self.pos.copy())
# 
#         distance_to_rso = np.linalg.norm(self.pos - self.target_pos)
#         radius_error = abs(distance_to_rso - self.desired_distance)
# 
#         in_viewing_shell = (
#             self.view_inner_radius <= distance_to_rso <= self.view_outer_radius
#         )
# 
#         in_crash_zone = distance_to_rso < self.crash_radius
# 
#         current_view_dir = self._get_current_view_dir()
#         angular_change = self._angle_between(prev_view_dir, current_view_dir)
# 
#         start_side_dot = float(np.dot(current_view_dir, self.start_view_dir))
#         new_opposite_side = False
# 
#         if (
#             not self.reached_opposite_side
#             and start_side_dot <= self.opposite_side_dot_threshold
#         ):
#             self.reached_opposite_side = True
#             new_opposite_side = True
# 
#         # Check which unvisited points are visible before marking them inspected.
#         visible_unvisited_before_marking = self._get_visible_unvisited_indices()
#         visible_signature = tuple(sorted(visible_unvisited_before_marking.tolist()))
# 
#         new_visible_patch = 0
# 
#         if self.prev_visible_signature is not None:
#             if visible_signature != self.prev_visible_signature and len(visible_signature) > 0:
#                 new_visible_patch = 1
# 
#         self.prev_visible_signature = visible_signature
# 
#         # Mark newly inspected points after computing visible patch signature.
#         newly_inspected = self._mark_inspected_points()
#         num_inspected = int(np.sum(self.inspected_mask))
# 
#         current_target_viewpoint = self._get_current_target_viewpoint()
# 
#         if current_target_viewpoint is not None:
#             current_target_distance = np.linalg.norm(current_target_viewpoint - self.pos)
#         else:
#             current_target_distance = 0.0
# 
#         reward = 0.0
# 
#         # Main reward: newly inspected surface points.
#         reward += self.new_point_reward * newly_inspected
# 
#         # Dense progress toward the desired viewpoint for the current target point.
#         if prev_target_viewpoint is not None and current_target_viewpoint is not None:
#             reward += self.viewpoint_progress_scale * (
#                 prev_target_distance - current_target_distance
#             )
# 
#         # Dense radial shaping: prefer useful viewing distance.
#         reward += self.radius_progress_scale * (
#             prev_radius_error - radius_error
#         )
# 
#         # Small reward for being in the valid viewing shell.
#         if in_viewing_shell:
#             reward += self.view_shell_reward
# 
#         # Exploration: reward angular motion around the RSO.
#         reward += self.angular_motion_reward_scale * angular_change
# 
#         # Exploration: bonus first time the deputy reaches the opposite side.
#         if new_opposite_side:
#             reward += self.opposite_side_reward
# 
#         # Exploration: reward looking at a different visible patch.
#         reward += self.new_visible_patch_reward * new_visible_patch
# 
#         # Small penalty if no new point was inspected.
#         if newly_inspected == 0:
#             reward += self.no_new_point_penalty
# 
#         # Penalize getting stuck at the same angular location without inspecting.
#         if angular_change < self.stuck_angle_threshold and newly_inspected == 0:
#             reward += self.stuck_penalty
# 
#         # Penalize staying on starting hemisphere too long.
#         if self.t > 100 and not self.reached_opposite_side:
#             if start_side_dot > 0.0:
#                 reward += self.same_hemisphere_penalty
# 
#         # Penalize control effort and high speed.
#         reward -= self.thrust_penalty_scale * np.linalg.norm(thrust_cmd)
#         reward -= self.speed_penalty_scale * np.linalg.norm(self.vel)
# 
#         # Penalize unsafe close approach before actual crash.
#         if distance_to_rso < self.view_inner_radius:
#             unsafe_depth = self.view_inner_radius - distance_to_rso
#             reward -= self.unsafe_zone_penalty_scale * unsafe_depth
# 
#         terminated = False
#         truncated = False
#         self.status = "Running"
# 
#         if in_crash_zone:
#             reward += self.crash_penalty
#             terminated = True
#             self.status = "Crash"
# 
#         elif num_inspected >= self.success_threshold:
#             reward += self.all_points_success_reward
#             terminated = True
#             self.status = "Success"
# 
#         elif distance_to_rso > self.max_distance:
#             reward += self.out_of_bounds_penalty
#             truncated = True
#             self.status = "Out of Bounds"
# 
#         elif self.t >= self.max_time:
#             truncated = True
#             self.status = "Timeout"
# 
#         obs = self._get_obs()
# 
#         info = {
#             "status": self.status,
#             "distance": distance_to_rso,
#             "radius_error": radius_error,
#             "thrust": thrust_cmd,
#             "velocity": self.vel.copy(),
#             "time": self.t,
#             "render_enabled": self.render_enabled,
#             "in_viewing_shell": in_viewing_shell,
#             "newly_inspected": newly_inspected,
#             "num_inspected": num_inspected,
#             "total_inspection_points": self.num_inspection_points,
#             "reached_opposite_side": self.reached_opposite_side,
#             "new_opposite_side": new_opposite_side,
#             "start_side_dot": start_side_dot,
#             "angular_change": angular_change,
#         }
# 
#         done = terminated or truncated
# 
#         if self.render_enabled:
#             self.render()
# 
#         return obs, reward, done, info
# 
#     def _get_obs(self):
#         obs = np.zeros(11, dtype=np.float32)
# 
#         obs[0:3] = self.pos
#         obs[3:6] = self.vel
#         obs[6] = self.sun_angle
# 
#         num_inspected = int(np.sum(self.inspected_mask))
#         obs[7] = np.float32(num_inspected)
# 
#         # Give the policy the RSO-centered direction of the current target
#         # surface point. This is analogous to nearest uninspected cluster
#         # direction in the real inspection environment.
#         target_point = self._get_current_target_point()
# 
#         if target_point is not None:
#             target_dir_from_rso = self._safe_unit(target_point - self.target_pos)
#             obs[8:11] = target_dir_from_rso
#         else:
#             obs[8:11] = np.zeros(3, dtype=np.float32)
# 
#         return obs

import os

import gym
import numpy as np
from gym.utils import seeding

import matplotlib

# Use non-interactive backend so this works without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class SimpleInspectionEnv(gym.Env):
    """
    Simplified single-inspector environment.

    Observation:
        [
            x, y, z,
            vx, vy, vz,
            sun_angle,
            num_inspected,
            target_dir_x, target_dir_y, target_dir_z
        ]

    Action:
        Discrete action mapped to 3D thrust [x, y, z].

    Task:
        Inspect discrete points on the RSO/chief surface.

    Camera assumption:
        The deputy camera always points toward the RSO/chief center.
        A point is inspected if it lies inside the camera viewing cone.

    Rendering:
        Rendering is OFF during training.
        train/oracle.py enables rendering only for the final deterministic rollout
        by calling env.enable_render(...).
    """

    def __init__(self, args=None):
        super().__init__()

        self.args = args

        # Rendering is OFF by default.
        self.render_enabled = False
        self.frame_dir = "frames"
        self.render_frame_idx = 0

        # Observation:
        # [x, y, z, vx, vy, vz, sun_angle, num_inspected, tx, ty, tz]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

        # 3 bins per axis -> 3^3 = 27 actions
        self.num_bins = 3
        self.action_space = gym.spaces.Discrete(self.num_bins ** 3)

        # Dynamics parameters
        self.dt = 1.0
        self.thrust_scale = 0.35
        self.velocity_damping = 0.96
        self.max_time = 500

        # RSO / chief position
        self.target_pos = np.zeros(3, dtype=np.float32)

        # Safety geometry
        self.crash_radius = 8.0
        self.max_distance = 220.0

        # Inspection/viewing shell
        self.success_inner_radius = 15.0
        self.success_outer_radius = 25.0

        # Put inspection points exactly on the inner sphere.
        self.inspection_point_radius = self.success_inner_radius 

        # Deputy should view those inner-sphere points from farther out.
        self.desired_distance = 20.0

        # Discrete inspection points on the RSO surface / inner sphere.
        self.num_inspection_points = 100
        self.success_threshold = self.num_inspection_points

        # Camera model:
        # The camera always points toward the RSO center.
        # A point is inspected if it lies inside this half-angle cone.
        self.camera_half_angle = np.deg2rad(15.0)

        # Only allow inspection while deputy is in this viewing shell.
        self.view_inner_radius = self.success_inner_radius
        self.view_outer_radius = 35.0

        # Reward magnitudes
        self.new_point_reward = 10.0
        self.all_points_success_reward = 200.0
        self.crash_penalty = -250.0
        self.out_of_bounds_penalty = -100.0

        # Dense shaping scales
        self.viewpoint_progress_scale = 1.5
        self.radius_progress_scale = 0.5
        self.view_shell_reward = 0.25

        # Penalties
        self.thrust_penalty_scale = 0.02
        self.speed_penalty_scale = 0.01
        self.unsafe_zone_penalty_scale = 1.0
        self.no_new_point_penalty = -0.01

        # logging
        self.episode_reward = 0.0
        self.episode_length = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def enable_render(self, frame_dir="frames", clear_old_frames=False):
        """
        Enable rendering for a single rollout.

        This should be called only after training finishes, from train/oracle.py.
        """
        self.render_enabled = True
        self.frame_dir = frame_dir
        self.render_frame_idx = 0

        if clear_old_frames and os.path.exists(self.frame_dir):
            import shutil
            shutil.rmtree(self.frame_dir)

        os.makedirs(self.frame_dir, exist_ok=True)

        print(f"[SimpleInspectionEnv] Rendering enabled. Saving frames to: {self.frame_dir}/")

    def disable_render(self):
        self.render_enabled = False

    def render(self, mode="human"):
        os.makedirs(self.frame_dir, exist_ok=True)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # RSO / chief
        ax.scatter(
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[2],
            s=140,
            label="RSO / Chief",
        )

        # Deputy
        ax.scatter(
            self.pos[0],
            self.pos[1],
            self.pos[2],
            s=90,
            label="Deputy",
        )

        # Inspection points
        visited_points = self.inspection_points[self.inspected_mask]
        unvisited_points = self.inspection_points[~self.inspected_mask]

        if len(unvisited_points) > 0:
            ax.scatter(
                unvisited_points[:, 0],
                unvisited_points[:, 1],
                unvisited_points[:, 2],
                s=12,
                alpha=0.45,
                label="Uninspected surface points",
            )

        if len(visited_points) > 0:
            ax.scatter(
                visited_points[:, 0],
                visited_points[:, 1],
                visited_points[:, 2],
                s=20,
                alpha=0.95,
                label="Inspected surface points",
            )

        # Current target surface point and desired viewing position
        target_point = self._get_current_target_point()
        target_viewpoint = self._get_current_target_viewpoint()

        if target_point is not None:
            ax.scatter(
                target_point[0],
                target_point[1],
                target_point[2],
                s=110,
                marker="x",
                label="Current surface target",
            )

        if target_viewpoint is not None:
            ax.scatter(
                target_viewpoint[0],
                target_viewpoint[1],
                target_viewpoint[2],
                s=90,
                marker="^",
                label="Desired viewing position",
            )

            ax.plot(
                [self.pos[0], target_viewpoint[0]],
                [self.pos[1], target_viewpoint[1]],
                [self.pos[2], target_viewpoint[2]],
                linestyle="--",
                linewidth=1.0,
                label="Path to viewing position",
            )

        # Trajectory
        if len(self.trajectory) > 1:
            traj = np.asarray(self.trajectory)
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                linewidth=1.5,
                label="Trajectory",
            )

        # Velocity vector
        ax.quiver(
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.vel[0],
            self.vel[1],
            self.vel[2],
            length=4.0,
            normalize=False,
            label="Velocity",
        )

        # Camera pointing direction: always toward RSO
        camera_forward = self._safe_unit(self.target_pos - self.pos)
        ax.quiver(
            self.pos[0],
            self.pos[1],
            self.pos[2],
            camera_forward[0],
            camera_forward[1],
            camera_forward[2],
            length=10.0,
            normalize=False,
            label="Camera forward",
        )

        # Radius spheres
        self._draw_sphere(ax, self.crash_radius, alpha=0.30)
        self._draw_sphere(ax, self.success_inner_radius, alpha=0.15)
        self._draw_sphere(ax, self.success_outer_radius, alpha=0.15)

        limit = 75
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        distance = np.linalg.norm(self.pos - self.target_pos)
        num_inspected = int(np.sum(self.inspected_mask))

        ax.set_title(
            f"frame={self.render_frame_idx}, "
            f"t={self.t}, "
            f"distance={distance:.2f}, "
            f"inspected={num_inspected}/{self.num_inspection_points}, "
            f"status={self.status}"
        )

        ax.legend(loc="upper right", fontsize=8)

        filename = os.path.join(
            self.frame_dir,
            f"frame_{self.render_frame_idx:04d}.png",
        )

        plt.savefig(filename)
        plt.close(fig)

        self.render_frame_idx += 1

    def _draw_sphere(self, ax, radius, alpha=0.25):
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)

        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(
            x + self.target_pos[0],
            y + self.target_pos[1],
            z + self.target_pos[2],
            alpha=alpha,
        )

    def _generate_inspection_points(self):
        """
        Generate approximately evenly spaced points on the inner sphere using
        the Fibonacci sphere method.
        """
        points = []

        n = self.num_inspection_points
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        for i in range(n):
            y = 1.0 - (2.0 * i) / float(n - 1)
            radius_xy = np.sqrt(max(0.0, 1.0 - y * y))

            theta = golden_angle * i

            x = np.cos(theta) * radius_xy
            z = np.sin(theta) * radius_xy

            unit_point = np.array([x, y, z], dtype=np.float32)

            # Points are placed on the inner sphere.
            point = self.target_pos + self.inspection_point_radius * unit_point
            points.append(point)

        return np.asarray(points, dtype=np.float32)

    def _map_action(self, action):
        """
        Converts one discrete action into [x_thrust, y_thrust, z_thrust].

        With num_bins = 3:
            values = [-1, 0, 1]
        """
        raw = np.array(
            np.unravel_index(
                int(action),
                (self.num_bins, self.num_bins, self.num_bins),
            )
        )

        values = np.linspace(-1.0, 1.0, self.num_bins, dtype=np.float32)
        return values[raw]

    def _safe_unit(self, vec, eps=1e-8):
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)

        if norm < eps:
            return np.zeros_like(vec, dtype=np.float32)

        return (vec / norm).astype(np.float32)

    def _get_unvisited_indices(self):
        return np.where(~self.inspected_mask)[0]

    def _get_current_target_index(self):
        """
        Pick the nearest unvisited inspection point to the deputy's desired
        viewing position, not necessarily nearest to the deputy itself.

        This gives the agent a local target direction to inspect next.
        """
        unvisited = self._get_unvisited_indices()

        if len(unvisited) == 0:
            return None

        unvisited_points = self.inspection_points[unvisited]
        distances = np.linalg.norm(unvisited_points - self.pos, axis=1)

        nearest_local_idx = int(np.argmin(distances))
        return int(unvisited[nearest_local_idx])

    def _get_current_target_point(self):
        idx = self._get_current_target_index()

        if idx is None:
            return None

        return self.inspection_points[idx]

    def _get_current_target_viewpoint(self):
        """
        Return a desired deputy viewpoint for the current unvisited inspection point.

        The inspection point lies on the inner sphere. The desired viewpoint is
        farther outward along the same radial direction.
        """
        target_point = self._get_current_target_point()

        if target_point is None:
            return None

        surface_dir = self._safe_unit(target_point - self.target_pos)
        viewpoint = self.target_pos + surface_dir * self.desired_distance

        return viewpoint.astype(np.float32)

    def _mark_inspected_points(self):
        """
        Mark unvisited inspection spots that fall inside the deputy camera cone.

        Assumption:
            The deputy camera is always pointing directly at the RSO/chief center.

        Geometry:
            camera_forward = direction from deputy to RSO center
            point_direction = direction from deputy to inspection point

        A point is inspected if:
            angle(camera_forward, point_direction) <= camera_half_angle
        """
        unvisited = self._get_unvisited_indices()

        if len(unvisited) == 0:
            return 0

        distance_to_rso = np.linalg.norm(self.pos - self.target_pos)

        # Only inspect while the deputy is in a reasonable viewing shell.
        in_viewing_shell = (
            self.view_inner_radius <= distance_to_rso <= self.view_outer_radius
        )

        if not in_viewing_shell:
            return 0

        camera_forward = self._safe_unit(self.target_pos - self.pos)

        unvisited_points = self.inspection_points[unvisited]
        vectors_to_points = unvisited_points - self.pos
        distances_to_points = np.linalg.norm(vectors_to_points, axis=1)

        valid = distances_to_points > 1e-8

        if not np.any(valid):
            return 0

        point_dirs = np.zeros_like(vectors_to_points, dtype=np.float32)
        point_dirs[valid] = vectors_to_points[valid] / distances_to_points[valid, None]

        cos_angles = point_dirs @ camera_forward
        cos_threshold = np.cos(self.camera_half_angle)

        visible_local = np.where(cos_angles >= cos_threshold)[0]

        if len(visible_local) == 0:
            return 0

        newly_seen_global = unvisited[visible_local]
        self.inspected_mask[newly_seen_global] = True

        return int(len(newly_seen_global))

    def reset(self):
        """
        Old Gym API reset.

        Stable-Baselines3 with older Gym expects:
            obs = env.reset()
        """
        self.pos = np.array([50.0, 0.0, 0.0], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)

        self.episode_reward = 0.0
        self.episode_length = 0

        self.sun_angle = 0.0
        self.t = 0
        self.status = "Running"

        self.inspection_points = self._generate_inspection_points()
        self.inspected_mask = np.zeros(self.num_inspection_points, dtype=bool)

        self.trajectory = [self.pos.copy()]

        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        thrust_cmd = self._map_action(action)

        prev_distance_to_rso = np.linalg.norm(self.pos - self.target_pos)
        prev_radius_error = abs(prev_distance_to_rso - self.desired_distance)

        prev_target_viewpoint = self._get_current_target_viewpoint()

        if prev_target_viewpoint is not None:
            prev_target_distance = np.linalg.norm(prev_target_viewpoint - self.pos)
        else:
            prev_target_distance = 0.0

        # Simple damped dynamics
        self.vel = (
            self.velocity_damping * self.vel
            + self.thrust_scale * thrust_cmd * self.dt
        )
        self.pos = self.pos + self.vel * self.dt

        self.t += 1
        self.trajectory.append(self.pos.copy())

        distance_to_rso = np.linalg.norm(self.pos - self.target_pos)
        radius_error = abs(distance_to_rso - self.desired_distance)

        in_viewing_shell = (
            self.view_inner_radius <= distance_to_rso <= self.view_outer_radius
        )

        in_crash_zone = distance_to_rso < self.crash_radius

        # Mark newly inspected points after moving.
        newly_inspected = self._mark_inspected_points()
        num_inspected = int(np.sum(self.inspected_mask))

        current_target_viewpoint = self._get_current_target_viewpoint()

        if current_target_viewpoint is not None:
            current_target_distance = np.linalg.norm(current_target_viewpoint - self.pos)
        else:
            current_target_distance = 0.0

        reward = 0.0

        # Main reward: number of newly inspected surface points.
        reward += self.new_point_reward * newly_inspected

        # Dense progress toward the desired viewpoint for the current target point.
        if prev_target_viewpoint is not None and current_target_viewpoint is not None:
            reward += self.viewpoint_progress_scale * (
                prev_target_distance - current_target_distance
            )

        # Dense radial shaping: prefer useful viewing distance.
        reward += self.radius_progress_scale * (
            prev_radius_error - radius_error
        )

        # Small reward for being in the valid viewing shell.
        if in_viewing_shell:
            reward += self.view_shell_reward

        # Small penalty if no new point was inspected.
        if newly_inspected == 0:
            reward += self.no_new_point_penalty

        # Penalize control effort and high speed.
        reward -= self.thrust_penalty_scale * np.linalg.norm(thrust_cmd)
        reward -= self.speed_penalty_scale * np.linalg.norm(self.vel)

        # Penalize unsafe close approach before actual crash.
        if distance_to_rso < self.view_inner_radius:
            unsafe_depth = self.view_inner_radius - distance_to_rso
            reward -= self.unsafe_zone_penalty_scale * unsafe_depth

        terminated = False
        truncated = False
        self.status = "Running"

        if in_crash_zone:
            reward += self.crash_penalty
            terminated = True
            self.status = "Crash"

        elif num_inspected >= self.success_threshold:
            reward += self.all_points_success_reward
            terminated = True
            self.status = "Success"

        elif distance_to_rso > self.max_distance:
            reward += self.out_of_bounds_penalty
            truncated = True
            self.status = "Out of Bounds"

        elif self.t >= self.max_time:
            truncated = True
            self.status = "Timeout"

        obs = self._get_obs()

        self.episode_length += 1
        self.episode_reward += float(reward)

        info = {
            "episode_reward": float(self.episode_reward),
            "episode_length": int(self.episode_length),
            "status": self.status,
            "distance": distance_to_rso,
            "radius_error": radius_error,
            "thrust": thrust_cmd,
            "velocity": self.vel.copy(),
            "time": self.t,
            "render_enabled": self.render_enabled,
            "in_viewing_shell": in_viewing_shell,
            "newly_inspected": newly_inspected,
            "num_inspected": num_inspected,
            "total_inspection_points": self.num_inspection_points,
            "current_target_distance": current_target_distance,
            "camera_half_angle_deg": float(np.rad2deg(self.camera_half_angle)),
            "inspection_point_radius": self.inspection_point_radius,
            "desired_distance": self.desired_distance,
        }

        done = terminated or truncated

        if self.render_enabled:
            self.render()

        return obs, reward, done, info

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)

        obs[0:3] = self.pos
        obs[3:6] = self.vel
        obs[6] = self.sun_angle

        num_inspected = int(np.sum(self.inspected_mask))
        obs[7] = np.float32(num_inspected)

        # Give the policy the RSO-centered direction of the nearest unvisited
        # surface point. This is analogous to the "nearest uninspected cluster"
        # direction in the real inspection environment.
        target_point = self._get_current_target_point()

        if target_point is not None:
            target_dir_from_rso = self._safe_unit(target_point - self.target_pos)
            obs[8:11] = target_dir_from_rso
        else:
            obs[8:11] = np.zeros(3, dtype=np.float32)

        return obs