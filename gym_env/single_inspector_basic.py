# import numpy as np
# import typing
# import gym
# import copy
# from safe_autonomy_simulation.sims import inspection as sim
# from . import single_inspector_reward as r
# from . import utils
# from gym.utils import seeding
# 
# class InspectionEnv(gym.Env):
#     def __init__(
#         self,
#         args,
#         success_threshold: float = 99,
#         crash_radius: float = 15,
#         max_distance: float = 800,
#         max_time: float = 12236,
#     ) -> None:
#         # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, theta_sun, n, x_ups, y_ups, z_ups]
#         self.observation_space = gym.spaces.Box(
#             np.concatenate(
#                 (
#                     [-np.inf] * 3,  # position
#                     [-np.inf] * 3,  # velocity
#                     [0],  # sun angle
#                     [0],  # num inspected
#                     [-1] * 3,  # nearest cluster
#                 )
#             ),
#             np.concatenate(
#                 (
#                     [np.inf] * 3,  # position
#                     [np.inf] * 3,  # velocity
#                     [2 * np.pi],  # sun angle
#                     [100],  # num inspected
#                     [1] * 3,  # nearest cluster
#                 )
#             ),
#             shape=(11,),
#         )
# 
#         self.args = args
#         self.num_bins = 21
#         self.action_space = gym.spaces.Discrete(self.num_bins ** 3)
# 
# 
#         # Environment parameters
#         self.crash_radius = crash_radius
#         self.max_distance = max_distance
#         self.max_time = max_time
#         self.success_threshold = success_threshold
# 
#         # Episode level information
#         self.prev_state = None
#         self.prev_num_inspected = 0
#         self.reward_components = {}
#         self.status = "Running"
#         self.seed()
# 
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
# 
#     def _map_action(self, action):
# 
#         raw_actions = np.array(np.unravel_index(
#             int(action),
#             (self.num_bins, self.num_bins, self.num_bins),
#             )
#         )
# 
#         values = np.linspace(-1.0, 1.0, self.num_bins, dtype=np.float32)
#         return values[raw_actions]
# 
#     def reset(
#         self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
#     ) -> tuple[typing.Any, dict[str, typing.Any]]:
#         #super().reset(seed=seed, options=options)
#         #super().reset()
#         self._init_sim()  # sim is light enough we just reconstruct it
#         self.simulator.reset()
#         self.reward_components = {}
#         self.status = "Running"
#         obs, info = self._get_obs(), self._get_info()
#         self.prev_state = None
#         self.prev_num_inspected = 0
#         # return obs, info
#         return obs
# 
#     def step(
#         self, action: typing.Any
#     ) -> tuple[typing.Any, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
#         assert self.action_space.contains(
#             action
#         ), f"given action {action} is not contained in action space {self.action_space}"
# 
#         # Remap the action space to [-1.0, 1.0] with 20 steps between
#         action = self._map_action(action)
# 
#         # Store previous simulator state
#         self.prev_state = self.sim_state.copy()
#         if self.simulator.sim_time > 0:
#             self.prev_num_inspected = (
#                 self.chief.inspection_points.get_num_points_inspected()
#             )
# 
#         # Update simulator state
#         self.deputy.add_control(action)
#         self.simulator.step()
# 
#         # Get info from simulator
#         observation = self._get_obs()
#         reward = self._get_reward()
#         terminated = self._get_terminated()
#         truncated = self._get_truncated()
#         done = terminated or truncated
#         info = self._get_info()
# 
#         if terminated or truncated:
#             print(
#                 "END:",
#                 self.status,
#                 "reward:",
#                 reward,
#                 "time:",
#                 self.simulator.sim_time,
#                 "num inspected:",
#                 self.chief.inspection_points.get_num_points_inspected(),
#                 "distance:",
#                 utils.rel_dist(
#                     pos1=self.chief.position,
#                     pos2=self.deputy.position,
#                 ),
#         )
# 
#         # print("OBS:", observation, type(observation), observation.shape)
# 
#         # return observation, reward, terminated, truncated, info
#         return observation, reward, done, info
# 
#     def _init_sim(self):
#         # Initialize spacecraft, sun, and simulator
#         self.chief = sim.Target(
#             name="chief",
#             num_points=100,
#             radius=10,
#         )
#         self.deputy = sim.Inspector(
#             name="deputy",
#             position=utils.polar_to_cartesian(
#                 r=self.np_random.uniform(50, 100),
#                 theta=self.np_random.uniform(0, 2 * np.pi),
#                 phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
#             ),
#             velocity=utils.polar_to_cartesian(
#                 r=self.np_random.uniform(0, 0.8),
#                 theta=self.np_random.uniform(0, 2 * np.pi),
#                 phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
#             ),
#             fov=np.pi,
#             focal_length=1,
#         )
#         self.sun = sim.Sun(theta=self.np_random.uniform(0, 2 * np.pi))
#         self.simulator = sim.InspectionSimulator(
#             frame_rate=0.1,
#             inspectors=[self.deputy],
#             targets=[self.chief],
#             sun=self.sun,
#         )
# 
#     # def _get_obs(self):
#     #     obs = self.observation_space.sample()
#     #     obs[:3] = self.deputy.position
#     #     obs[3:6] = self.deputy.velocity
#     #     obs[6] = self.sun.theta % (2 * np.pi)
#     #     obs[7] = self.chief.inspection_points.get_num_points_inspected()
#     #     obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(
#     #         camera=self.deputy.camera, sun=self.sun
#     #     )
#     #     return obs
# 
#     def _get_obs(self):
#         obs = np.zeros(11, dtype=np.float32)
#         obs[:3] = np.asarray(self.deputy.position, dtype=np.float32)
#         obs[3:6] = np.asarray(self.deputy.velocity, dtype=np.float32)
#         obs[6] = np.float32(self.sun.theta % (2 * np.pi))
#         obs[7] = np.float32(self.chief.inspection_points.get_num_points_inspected())
# 
#         cluster = self.chief.inspection_points.kmeans_find_nearest_cluster(
#             camera=self.deputy.camera,
#             sun=self.sun,
#         )
#         obs[8:11] = np.asarray(cluster, dtype=np.float32)
# 
#         return obs
# 
#     def _get_info(self):
#         return {"reward_components": copy.copy(self.reward_components), "status": copy.copy(self.status),
#                 "sim_time": self.simulator.sim_time,
#                 "num_inspected": self.chief.inspection_points.get_num_points_inspected(),
#                 "distance": utils.rel_dist(
#                     pos1=self.chief.position,
#                     pos2=self.deputy.position,
#             ),
#         }
# 
#     def _get_reward(self):
#         reward = 0
# 
#         # Dense rewards
#         points_reward = r.observed_points_reward(
#             chief=self.chief, prev_num_inspected=self.prev_num_inspected
#         )
#         self.reward_components["observed_points"] = points_reward
#         reward += points_reward
# 
#         delta_v_reward = r.delta_v_reward(
#             v=self.deputy.velocity,
#             prev_v=self.prev_state["deputy"][3:6],
#         )
#         self.reward_components["delta_v"] = delta_v_reward
#         reward += delta_v_reward
# 
#         # Sparse rewards
#         success_reward = r.inspection_success_reward(
#             chief=self.chief,
#             total_points=self.success_threshold,
#         )
#         self.reward_components["success"] = success_reward
#         reward += success_reward
# 
#         crash_reward = r.crash_reward(
#             chief=self.chief,
#             deputy=self.deputy,
#             crash_radius=self.crash_radius,
#         )
#         self.reward_components["crash"] = crash_reward
#         reward += crash_reward
# 
#         # TODO: add another reward based on how long simulation runs
# 
#         return reward
# 
#     def _get_terminated(self):
#         # Get state info
#         d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)
# 
#         # Determine if in terminal state
#         crash = d < self.crash_radius
#         all_inspected = (
#             self.chief.inspection_points.get_num_points_inspected()
#             >= self.success_threshold
#         )
# 
#         # Update Status
#         if crash:
#             self.status = "Crash"
#         elif all_inspected:
#             self.status = "Success"
# 
#         return crash or all_inspected
# 
#     def _get_truncated(self):
#         d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)
#         timeout = self.simulator.sim_time > self.max_time
#         oob = d > self.max_distance
# 
#         # Update Status
#         if oob:
#            self.status = "Out of Bounds"
#         elif timeout:
#             self.status = "Timeout"
# 
#         return timeout or oob
# 
#     @property
#     def sim_state(self) -> dict:
#         state = {
#             "deputy": self.deputy.state,
#             "chief": self.chief.state,
#         }
#         return state

import os
import typing
import copy

import gym
import numpy as np
from gym.utils import seeding

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_autonomy_simulation.sims import inspection as sim
from . import single_inspector_reward as r
from . import utils


class InspectionEnv(gym.Env):
    def __init__(
        self,
        args,
        success_threshold: float = 99,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 12236,
    ) -> None:

        self.args = args

        # Rendering is OFF by default.
        # train/oracle.py should enable it only for the final deterministic rollout.
        self.render_enabled = False
        self.frame_dir = "frames"
        self.render_frame_idx = 0

        # Each spacecraft obs =
        # [x, y, z, v_x, v_y, v_z, theta_sun, n, x_ups, y_ups, z_ups]
        self.observation_space = gym.spaces.Box(
            np.concatenate(
                (
                    [-np.inf] * 3,  # position
                    [-np.inf] * 3,  # velocity
                    [0],            # sun angle
                    [0],            # num inspected
                    [-1] * 3,       # nearest cluster
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,       # position
                    [np.inf] * 3,       # velocity
                    [2 * np.pi],        # sun angle
                    [100],              # num inspected
                    [1] * 3,            # nearest cluster
                )
            ),
            shape=(11,),
            dtype=np.float32,
        )

        self.num_bins = 21
        self.action_space = gym.spaces.Discrete(self.num_bins ** 3)

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.prev_num_inspected = 0
        self.reward_components = {}
        self.status = "Running"

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def enable_render(self, frame_dir="frames", clear_old_frames=False):
        """
        Enable rendering for a single rollout.

        This should be called from train/oracle.py after training finishes,
        not during vectorized training.
        """
        self.render_enabled = True
        self.frame_dir = frame_dir
        self.render_frame_idx = 0

        if clear_old_frames and os.path.exists(self.frame_dir):
            import shutil
            shutil.rmtree(self.frame_dir)

        os.makedirs(self.frame_dir, exist_ok=True)

    def disable_render(self):
        self.render_enabled = False

    def render(self, mode="human"):
        """
        Save a 3D frame of the current inspection state.

        Shows:
        - chief/target at origin
        - deputy/inspector position
        - deputy velocity vector
        - crash radius
        - max-distance boundary
        - nearest uninspected cluster direction/location from observation
        """
        os.makedirs(self.frame_dir, exist_ok=True)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        chief_pos = np.asarray(self.chief.position, dtype=np.float32)
        deputy_pos = np.asarray(self.deputy.position, dtype=np.float32)
        deputy_vel = np.asarray(self.deputy.velocity, dtype=np.float32)

        # Chief / target
        ax.scatter(
            chief_pos[0],
            chief_pos[1],
            chief_pos[2],
            s=120,
            label="Chief / Target",
        )

        # Deputy / inspector
        ax.scatter(
            deputy_pos[0],
            deputy_pos[1],
            deputy_pos[2],
            s=80,
            label="Deputy / Inspector",
        )

        # Deputy velocity vector
        ax.quiver(
            deputy_pos[0],
            deputy_pos[1],
            deputy_pos[2],
            deputy_vel[0],
            deputy_vel[1],
            deputy_vel[2],
            length=10.0,
            normalize=False,
            label="Velocity",
        )

        # Draw crash radius around chief
        self._draw_sphere(ax, chief_pos, self.crash_radius, alpha=0.18)

        # Draw max distance boundary around chief
        self._draw_sphere(ax, chief_pos, self.max_distance, alpha=0.05)

        # Draw nearest cluster direction/point from observation
        try:
            cluster = self.chief.inspection_points.kmeans_find_nearest_cluster(
                camera=self.deputy.camera,
                sun=self.sun,
            )
            cluster = np.asarray(cluster, dtype=np.float32)

            # In your obs, cluster appears to be normalized-ish [-1, 1],
            # so scale it visually around the chief radius.
            cluster_point = chief_pos + cluster * 25.0

            ax.scatter(
                cluster_point[0],
                cluster_point[1],
                cluster_point[2],
                s=60,
                marker="x",
                label="Nearest Cluster",
            )

            ax.plot(
                [chief_pos[0], cluster_point[0]],
                [chief_pos[1], cluster_point[1]],
                [chief_pos[2], cluster_point[2]],
                linestyle="--",
            )
        except Exception:
            # Rendering should never crash training/testing.
            pass

        distance = utils.rel_dist(
            pos1=self.chief.position,
            pos2=self.deputy.position,
        )

        num_inspected = self.chief.inspection_points.get_num_points_inspected()

        ax.set_title(
            f"t={self.simulator.sim_time:.1f}, "
            f"status={self.status}, "
            f"distance={distance:.2f}, "
            f"inspected={num_inspected}"
        )

        limit = max(self.max_distance * 1.05, 100.0)
        ax.set_xlim(chief_pos[0] - limit, chief_pos[0] + limit)
        ax.set_ylim(chief_pos[1] - limit, chief_pos[1] + limit)
        ax.set_zlim(chief_pos[2] - limit, chief_pos[2] + limit)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()

        filename = os.path.join(
            self.frame_dir,
            f"frame_{self.render_frame_idx:04d}.png",
        )

        plt.savefig(filename)
        plt.close(fig)

        self.render_frame_idx += 1

    def _draw_sphere(self, ax, center, radius, alpha=0.15):
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)

        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        ax.plot_wireframe(x, y, z, alpha=alpha)

    def _map_action(self, action):
        raw_actions = np.array(
            np.unravel_index(
                int(action),
                (self.num_bins, self.num_bins, self.num_bins),
            )
        )

        values = np.linspace(-1.0, 1.0, self.num_bins, dtype=np.float32)
        return values[raw_actions]

    def reset(self):
        """
        Old Gym API reset.

        Stable-Baselines3 expects:
            obs = env.reset()
        """
        self._init_sim()
        self.simulator.reset()

        self.reward_components = {}
        self.status = "Running"

        self.prev_state = None
        self.prev_num_inspected = 0

        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"given action {action} is not contained in action space {self.action_space}"

        # Remap the action space to [-1.0, 1.0]
        action = self._map_action(action)

        # Store previous simulator state
        self.prev_state = self.sim_state.copy()

        if self.simulator.sim_time > 0:
            self.prev_num_inspected = (
                self.chief.inspection_points.get_num_points_inspected()
            )

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        done = terminated or truncated
        info = self._get_info()

        if terminated or truncated:
            print(
                "END:",
                self.status,
                "reward:",
                reward,
                "time:",
                self.simulator.sim_time,
                "num inspected:",
                self.chief.inspection_points.get_num_points_inspected(),
                "distance:",
                utils.rel_dist(
                    pos1=self.chief.position,
                    pos2=self.deputy.position,
                ),
            )

        if self.render_enabled:
            self.render()

        return observation, reward, done, info

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        self.chief = sim.Target(
            name="chief",
            num_points=100,
            radius=10,
        )

        self.deputy = sim.Inspector(
            name="deputy",
            position=utils.polar_to_cartesian(
                r=self.np_random.uniform(50, 100),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            velocity=utils.polar_to_cartesian(
                r=self.np_random.uniform(0, 0.8),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            fov=np.pi,
            focal_length=1,
        )

        self.sun = sim.Sun(theta=self.np_random.uniform(0, 2 * np.pi))

        self.simulator = sim.InspectionSimulator(
            frame_rate=0.1,
            inspectors=[self.deputy],
            targets=[self.chief],
            sun=self.sun,
        )

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)

        obs[:3] = np.asarray(self.deputy.position, dtype=np.float32)
        obs[3:6] = np.asarray(self.deputy.velocity, dtype=np.float32)
        obs[6] = np.float32(self.sun.theta % (2 * np.pi))
        obs[7] = np.float32(
            self.chief.inspection_points.get_num_points_inspected()
        )

        cluster = self.chief.inspection_points.kmeans_find_nearest_cluster(
            camera=self.deputy.camera,
            sun=self.sun,
        )

        obs[8:11] = np.asarray(cluster, dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "reward_components": copy.copy(self.reward_components),
            "status": copy.copy(self.status),
            "sim_time": self.simulator.sim_time,
            "num_inspected": self.chief.inspection_points.get_num_points_inspected(),
            "distance": utils.rel_dist(
                pos1=self.chief.position,
                pos2=self.deputy.position,
            ),
        }

    def _get_reward(self):
        reward = 0.0

        # Dense rewards
        points_reward = r.observed_points_reward(
            chief=self.chief,
            prev_num_inspected=self.prev_num_inspected,
        )
        self.reward_components["observed_points"] = points_reward
        reward += points_reward

        delta_v_reward = r.delta_v_reward(
            v=self.deputy.velocity,
            prev_v=self.prev_state["deputy"][3:6],
        )
        self.reward_components["delta_v"] = delta_v_reward
        reward += delta_v_reward

        # Sparse rewards
        success_reward = r.inspection_success_reward(
            chief=self.chief,
            total_points=self.success_threshold,
        )
        self.reward_components["success"] = success_reward
        reward += success_reward

        crash_reward = r.crash_reward(
            chief=self.chief,
            deputy=self.deputy,
            crash_radius=self.crash_radius,
        )
        self.reward_components["crash"] = crash_reward
        reward += crash_reward

        return reward

    def _get_terminated(self):
        d = utils.rel_dist(
            pos1=self.chief.position,
            pos2=self.deputy.position,
        )

        crash = d < self.crash_radius

        all_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
            >= self.success_threshold
        )

        if crash:
            self.status = "Crash"
        elif all_inspected:
            self.status = "Success"

        return crash or all_inspected

    def _get_truncated(self):
        d = utils.rel_dist(
            pos1=self.chief.position,
            pos2=self.deputy.position,
        )

        timeout = self.simulator.sim_time > self.max_time
        oob = d > self.max_distance

        if oob:
            self.status = "Out of Bounds"
        elif timeout:
            self.status = "Timeout"

        return timeout or oob

    @property
    def sim_state(self) -> dict:
        return {
            "deputy": self.deputy.state,
            "chief": self.chief.state,
        }
