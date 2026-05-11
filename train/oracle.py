# from stable_baselines3 import DQN, PPO
# 
# from gym_env import make_env
# from model.paths import get_oracle_path
# from train.learning_rates import LinearSchedule, HalfLinearSchedule
# 
# 
# def train_oracle(args):
#     env = make_env(args)
#     if args.resume:
#         print("Resuming training")
#         cls, policy_kwargs = get_model_cls(args)
#         model = cls.load(get_oracle_path(args), env=env)
#     else:
#         model, policy_kwargs = get_model(env, args)
# 
#     log_name = f"{args.log_prefix}{args.env_name}_{args.n_env}env_{kwargs_to_str(policy_kwargs)}"
# 
#     model.learn(total_timesteps=args.total_timesteps, eval_freq=args.total_timesteps // 10,
#                 reset_num_timesteps=not args.resume, tb_log_name=log_name)
#     model_path = get_oracle_path(args)
#     model.save(model_path)
#     model.save(f"./log/{log_name}/model")
#     print(f"Training complete. Saved model to {model_path}")
# 
# 
# # DQN requires specific hyperparameter tuning
# # taken from here: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
# ENV_TO_MODEL = {
#     'PongNoFrameskip-v4': {
#         'model': DQN,
#         'kwargs': {
#             'policy': 'CnnPolicy',
#             'learning_starts': 100000,
#             'learning_rate': 1e-4,
#             'buffer_size': 1_000_000,
#             'batch_size': 32,
#             'target_update_interval': 1000,
#             'train_freq': 4,
#             'gradient_steps': 1,
#             'exploration_fraction': 0.1,
#             'exploration_final_eps': 0.01,
#             'optimize_memory_usage': True,
#         },
#         # args we resume with
#         'kwargs_resume': {
#             'exploration_final_eps': 0.01,
#             'exploration_initial_eps': 0.1
#         }
#     },
#     'CartPole-v1': {
#         'model': PPO,
#         'kwargs': {
#             'policy': 'MlpPolicy',
#             'batch_size': 256,
#             'n_steps': 32,
#             'n_epochs': 20,
#             'learning_rate': LinearSchedule(0.001),
#             'clip_range': LinearSchedule(0.2),
#             'gae_lambda': 0.8,
#             'gamma': 0.98,
#             'ent_coef': 0.0
#         }
#     },
#     'ToyPong-v0': {
#         'model': PPO,
#         'kwargs': {
#             'policy': 'MlpPolicy',
#             'learning_rate': HalfLinearSchedule(0.0003),
#             'policy_kwargs': {
#                 'net_arch': [64, dict(pi=[128, 64], vf=[64, 64])]
#             }
#         }
#     },
#     'SingleInspector-v0': {
#         'model': PPO,
#         'kwargs': {
#             'policy': 'MlpPolicy',
#             'batch_size': 256,
#             'n_steps': 32,
#             'n_epochs': 20,
#             'learning_rate': LinearSchedule(0.001),
#             'clip_range': LinearSchedule(0.2),
#             'gae_lambda': 0.8,
#             'gamma': 0.98,
#             'ent_coef': 0.0
#         }
#     },
#     'SimpleSingleInspector-v0': {
#         'model': PPO,
#         'kwargs': {
#             'policy': 'MlpPolicy',
#             'batch_size': 256,
#             'n_steps': 32,
#             'n_epochs': 20,
#             'learning_rate': LinearSchedule(0.001),
#             'clip_range': LinearSchedule(0.2),
#             'gae_lambda': 0.8,
#             'gamma': 0.98,
#             'ent_coef': 0.0
#         }
#     },
# #    'SingleInspector-v0': {
# #        'model': PPO,
# #        'kwargs': {
# #            'policy': 'MlpPolicy',
# #            'batch_size': 256,
# #            'n_steps': 256,
# #            'n_epochs': 5,
# #            'learning_rate': 3e-4,
# #            'clip_range': LinearSchedule(0.2),
# #            'gae_lambda': 0.95,
# #            'gamma': 0.995,
# #            'ent_coef': 0.0
# #        }
# #    },
# }
# 
# 
# def kwargs_to_str(kwargs):
#     return '_'.join([f"{k}-{v}" for k, v in kwargs.items() if k not in ['policy', 'policy_kwargs']])
# 
# 
# def get_model_cls(args):
#     if args.env_name not in ENV_TO_MODEL:
#         raise ValueError(f"Unsupported env: {args.env_name}")
# 
#     return ENV_TO_MODEL[args.env_name]['model'], ENV_TO_MODEL[args.env_name]['kwargs']
# 
# 
# def get_model(env, args):
#     if args.env_name not in ENV_TO_MODEL:
#         raise ValueError(f"Unsupported env: {args.env_name}")
# 
#     cfg = ENV_TO_MODEL[args.env_name]
#     model_kwargs = cfg['kwargs']
#     if args.resume:
#         model_kwargs.update(cfg['kwargs_resume'])
#     return cfg['model'](env=env, verbose=args.verbose, tensorboard_log='./log', seed=args.seed,
#                         **model_kwargs), model_kwargs
# 


import os
import shutil
import subprocess

import gym
from stable_baselines3 import DQN, PPO

from gym_env import make_env
from model.paths import get_oracle_path
from train.learning_rates import LinearSchedule, HalfLinearSchedule


def train_oracle(args):
    env = make_env(args)

    if args.resume:
        print("Resuming training")
        cls, policy_kwargs = get_model_cls(args)
        model = cls.load(get_oracle_path(args), env=env)
    else:
        model, policy_kwargs = get_model(env, args)

    log_name = f"{args.log_prefix}{args.env_name}_{args.n_env}env_{kwargs_to_str(policy_kwargs)}"

    model.learn(
        total_timesteps=args.total_timesteps,
        eval_freq=args.total_timesteps // 10,
        reset_num_timesteps=not args.resume,
        tb_log_name=log_name,
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


def make_render_video(frame_dir="frames"):
    """
    Create a Windows/VS Code compatible MP4 from saved frames.
    """

    output_path = "simple_env_final_rollout.mp4"

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