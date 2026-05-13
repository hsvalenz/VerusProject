import os
import shutil
import subprocess

import gym
from stable_baselines3 import DQN, PPO

from gym_env import make_env
from model.paths import get_oracle_path
from train.learning_rates import LinearSchedule, HalfLinearSchedule

from stable_baselines3.common.callbacks import BaseCallback

class InspectionTensorboardCallback(BaseCallback):
    """
    Logs useful environment info values to TensorBoard.

    This expects the environment's step() function to return an info dict
    containing fields like:
        distance, radius_error, newly_inspected, num_inspected, status, etc.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        if not infos:
            return True

        distances = []
        radius_errors = []
        newly_inspected_values = []
        num_inspected_values = []
        current_target_distances = []
        episode_rewards = []
        episode_lengths = []

        # Fuel metrics
        fuel_used_this_step_values = []
        total_fuel_used_values = []
        fuel_remaining_values = []
        fuel_remaining_fraction_values = []

        success_count = 0
        crash_count = 0
        out_of_bounds_count = 0
        timeout_count = 0
        out_of_fuel_count = 0
        in_viewing_shell_count = 0

        thrust_norms = []
        velocity_norms = []

        for info in infos:
            if "distance" in info:
                distances.append(float(info["distance"]))

            if "radius_error" in info:
                radius_errors.append(float(info["radius_error"]))

            if "newly_inspected" in info:
                newly_inspected_values.append(float(info["newly_inspected"]))

            if "num_inspected" in info:
                num_inspected_values.append(float(info["num_inspected"]))

            if "current_target_distance" in info:
                current_target_distances.append(float(info["current_target_distance"]))

            if "episode_reward" in info:
                episode_rewards.append(float(info["episode_reward"]))

            if "episode_length" in info:
                episode_lengths.append(float(info["episode_length"]))
            
            # Fuel logging values from the env info dict
            if "fuel_used_this_step" in info:
                fuel_used_this_step_values.append(float(info["fuel_used_this_step"]))

            if "total_fuel_used" in info:
                total_fuel_used_values.append(float(info["total_fuel_used"]))

            if "fuel_remaining" in info:
                fuel_remaining_values.append(float(info["fuel_remaining"]))

            if "fuel_remaining_fraction" in info:
                fuel_remaining_fraction_values.append(float(info["fuel_remaining_fraction"]))

            if info.get("in_viewing_shell", False):
                in_viewing_shell_count += 1

            status = info.get("status", "")

            if status == "Success" or info.get("success", False):
                success_count += 1

            if status == "Crash" or info.get("crash", False):
                crash_count += 1

            if status == "Out of Bounds" or info.get("out_of_bounds", False):
                out_of_bounds_count += 1

            if status == "Timeout" or info.get("timeout", False):
                timeout_count += 1

            if "thrust" in info:
                thrust = info["thrust"]
                thrust_norms.append(float((thrust ** 2).sum() ** 0.5))

            if "velocity" in info:
                velocity = info["velocity"]
                velocity_norms.append(float((velocity ** 2).sum() ** 0.5))

        n_envs = max(len(infos), 1)

        def record_mean(name, values):
            if values:
                self.logger.record(name, sum(values) / len(values))

        record_mean("inspection/distance_mean", distances)
        record_mean("inspection/radius_error_mean", radius_errors)
        record_mean("inspection/newly_inspected_mean", newly_inspected_values)
        record_mean("inspection/num_inspected_mean", num_inspected_values)
        record_mean("inspection/current_target_distance_mean", current_target_distances)
        record_mean("inspection/episode_reward_running", episode_rewards)
        record_mean("inspection/episode_length_running", episode_lengths)
        record_mean("inspection/thrust_norm_mean", thrust_norms)
        record_mean("inspection/velocity_norm_mean", velocity_norms)

        # Fuel TensorBoard logs
        record_mean("fuel/fuel_used_this_step_mean", fuel_used_this_step_values)
        record_mean("fuel/total_fuel_used_mean", total_fuel_used_values)
        record_mean("fuel/fuel_remaining_mean", fuel_remaining_values)
        record_mean("fuel/fuel_remaining_fraction_mean", fuel_remaining_fraction_values)

        self.logger.record("inspection/in_viewing_shell_fraction", in_viewing_shell_count / n_envs)
        self.logger.record("inspection/success_fraction", success_count / n_envs)
        self.logger.record("inspection/crash_fraction", crash_count / n_envs)
        self.logger.record("inspection/out_of_bounds_fraction", out_of_bounds_count / n_envs)
        self.logger.record("inspection/timeout_fraction", timeout_count / n_envs)

        return True

def train_oracle(args):
    env = make_env(args)

    if args.resume:
        print("Resuming training")
        cls, policy_kwargs = get_model_cls(args)
        model = cls.load(get_oracle_path(args), env=env)
    else:
        model, policy_kwargs = get_model(env, args)

    log_name = f"{args.log_prefix}{args.env_name}_{args.n_env}env_{kwargs_to_str(policy_kwargs)}"

    callback = InspectionTensorboardCallback(verbose=args.verbose)

    model.learn(
        total_timesteps=args.total_timesteps,
        eval_freq=args.total_timesteps // 10,
        reset_num_timesteps=not args.resume,
        tb_log_name=log_name,
        callback=callback,
        log_interval=1,
    )

    model_path = get_oracle_path(args)
    model.save(model_path)
    model.save(f"./log/{log_name}/model")

    print(f"Training complete. Saved model to {model_path}")

    if args.render:
        render_final_deterministic_rollout(model, args)


def render_final_deterministic_rollout(model, args):
    """
    Run one clean deterministic rollout after training finishes.

    This uses a single non-vectorized Gym env so the frames correspond to one
    episode from the final trained policy.
    """

    frame_dir = "frames"

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)

    os.makedirs(frame_dir, exist_ok=True)

    print("Running final deterministic rollout...")

    env = gym.make(args.env_name, args=args)

    # This assumes your environment has enable_render().
    # If it does not, it will fall back to setting render_enabled directly.
    if hasattr(env, "enable_render"):
        env.enable_render(frame_dir=frame_dir, clear_old_frames=False)
    else:
        env.render_enabled = True
        env.frame_dir = frame_dir

    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    final_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        # If your env renders automatically when render_enabled=True,
        # this explicit call is not needed. This fallback makes it robust.
        if not getattr(env, "render_enabled", False):
            env.render()

        total_reward += float(reward)
        step += 1
        final_info = info

    print("Final deterministic rollout complete.")
    print(f"Episode reward: {total_reward}")
    print(f"Episode steps: {step}")
    print(f"Final info: {final_info}")

    make_render_video(frame_dir=frame_dir)


def make_render_video(frame_dir="frames", output_path=None):
    """
    Create a Windows/VS Code compatible MP4 from saved frames.

    If output_path is not provided, saves a uniquely named video in ./log/videos/.
    """

    from datetime import datetime

    video_dir = "./log/videos"
    os.makedirs(video_dir, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            video_dir,
            f"simple_env_final_rollout_{timestamp}.mp4"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "10",
        "-i",
        f"{frame_dir}/frame_%04d.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path,
    ]

    print("Creating final rollout video...")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"Saved final rollout video to: {output_path}")
    else:
        print("ffmpeg failed to create the video.")
        print("You can still inspect the saved PNG frames in the frames/ directory.")


# DQN requires specific hyperparameter tuning
# taken from here: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
ENV_TO_MODEL = {
    'PongNoFrameskip-v4': {
        'model': DQN,
        'kwargs': {
            'policy': 'CnnPolicy',
            'learning_starts': 100000,
            'learning_rate': 1e-4,
            'buffer_size': 1_000_000,
            'batch_size': 32,
            'target_update_interval': 1000,
            'train_freq': 4,
            'gradient_steps': 1,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.01,
            'optimize_memory_usage': True,
        },
        'kwargs_resume': {
            'exploration_final_eps': 0.01,
            'exploration_initial_eps': 0.1,
        },
    },
    'CartPole-v1': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 256,
            'n_steps': 32,
            'n_epochs': 20,
            'learning_rate': LinearSchedule(0.001),
            'clip_range': LinearSchedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.0,
        },
    },
    'ToyPong-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'learning_rate': HalfLinearSchedule(0.0003),
            'policy_kwargs': {
                'net_arch': [64, dict(pi=[128, 64], vf=[64, 64])],
            },
        },
    },
    'SingleInspector-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 256,
            'n_steps': 32,
            'n_epochs': 20,
            'learning_rate': LinearSchedule(0.001),
            'clip_range': LinearSchedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.0,
        },
    },
    'SimpleSingleInspector-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 256,
            'n_steps': 32,
            'n_epochs': 20,
            'learning_rate': LinearSchedule(0.001),
            #'learning_rate': HalfLinearSchedule(1e-3),
            'clip_range': LinearSchedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.01,
        },
    },
    'FuelSingleInspector-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 256,
            'n_steps': 32,
            'n_epochs': 20,
            'learning_rate': LinearSchedule(0.001),
            #'learning_rate': HalfLinearSchedule(1e-3),
            'clip_range': LinearSchedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.01,
        },
    },
}


def kwargs_to_str(kwargs):
    return '_'.join(
        [
            f"{k}-{v}"
            for k, v in kwargs.items()
            if k not in ['policy', 'policy_kwargs']
        ]
    )


def get_model_cls(args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    return ENV_TO_MODEL[args.env_name]['model'], ENV_TO_MODEL[args.env_name]['kwargs']


def get_model(env, args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    cfg = ENV_TO_MODEL[args.env_name]

    # Make a copy so resume updates do not mutate ENV_TO_MODEL globally.
    model_kwargs = dict(cfg['kwargs'])

    if args.resume and 'kwargs_resume' in cfg:
        model_kwargs.update(cfg['kwargs_resume'])

    return (
        cfg['model'](
            env=env,
            verbose=args.verbose,
            tensorboard_log='./log',
            seed=args.seed,
            **model_kwargs,
        ),
        model_kwargs,
    )