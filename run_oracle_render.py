import os
import shutil
import subprocess
import argparse

import gym
from stable_baselines3 import PPO, DQN

# Important: import/register your custom gym envs
import gym_env


def load_model(model_path, algo, env):
    algo = algo.lower()

    if algo == "ppo":
        return PPO.load(model_path, env=env)

    if algo == "dqn":
        return DQN.load(model_path, env=env)

    raise ValueError(f"Unsupported algo: {algo}. Use 'ppo' or 'dqn'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-name", type=str, default="SimpleSingleInspector-v0")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--frame-dir", type=str, default="frames")
    parser.add_argument("--video-path", type=str, default="final_rollout.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    # Clear old frames
    if os.path.exists(args.frame_dir):
        shutil.rmtree(args.frame_dir)

    os.makedirs(args.frame_dir, exist_ok=True)

    # Create a normal, non-vectorized env
    env = gym.make(args.env_name, args=None)

    # Enable rendering if your env supports it
    if hasattr(env, "enable_render"):
        env.enable_render(frame_dir=args.frame_dir, clear_old_frames=False)
    else:
        env.render_enabled = True
        env.frame_dir = args.frame_dir

    model = load_model(args.model_path, args.algo, env)

    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    final_info = {}

    print("Initial obs:", obs)

    while not done:
        action, _ = model.predict(obs, deterministic=args.deterministic)

        print(f"\nBefore step {step + 1}")
        print("action:", action)

        obs, reward, done, info = env.step(action)

        total_reward += float(reward)
        step += 1
        final_info = info

       # print("reward:", reward)
       # print("done:", done)
       # print("info:", info)

        if step >= 1000:
            print("Stopping manually at 1000 steps")
            break

    print("\nRollout complete")
    print(f"Total reward: {total_reward}")
    print(f"Steps: {step}")
    print(f"Final info: {final_info}")

    # Make compatible MP4
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        os.path.join(args.frame_dir, "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        args.video_path,
    ]

    print("\nCreating video...")
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"Saved video to: {args.video_path}")
    else:
        print("ffmpeg failed. The PNG frames should still be in:", args.frame_dir)


if __name__ == "__main__":
    main()