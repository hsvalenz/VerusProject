import argparse
import os
import shutil
import subprocess

import gym
import joblib
import numpy as np

import gym_env

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

######################################################################
# Saving tree as image
def get_observation_feature_names():
    """
    Names for the 11 observations used by SimpleInspectionEnv.

    Observation:
        [
            x, y, z,
            vx, vy, vz,
            sun_angle,
            num_inspected,
            target_dir_x, target_dir_y, target_dir_z
        ]
    """
    return [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "sun_angle",
        "num_inspected",
        "target_dir_x",
        "target_dir_y",
        "target_dir_z",
        "fuel_remaining_fraction",
    ]


def make_action_names(num_bins=3):
    """
    Create readable names for the 27 discrete thrust actions.

    The environment maps action -> [x_thrust, y_thrust, z_thrust]
    using np.unravel_index(action, (3, 3, 3)).
    """
    values = np.linspace(-1.0, 1.0, num_bins, dtype=np.float32)

    action_names = []

    for action in range(num_bins ** 3):
        ix, iy, iz = np.unravel_index(
            action,
            (num_bins, num_bins, num_bins),
        )

        thrust_x = values[ix]
        thrust_y = values[iy]
        thrust_z = values[iz]

        action_names.append(
            f"a{action}: thrust=[{thrust_x:.0f},{thrust_y:.0f},{thrust_z:.0f}]"
        )

    return action_names


def save_tree_image(policy, output_path="viper_tree.png", max_depth=None):
    """
    Save the sklearn decision tree as a PNG image with meaningful feature
    and action labels.
    """

    feature_names = get_observation_feature_names()
    action_names = make_action_names(num_bins=3)

    plt.figure(figsize=(28, 16))

    plot_tree(
        policy,
        feature_names=feature_names,
        class_names=action_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        max_depth=max_depth,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved tree image to: {output_path}")


def make_side_by_side_frames(
    sim_frame_dir="viper_output/viper_frames",
    tree_image_path="viper_output/viper_tree.png",
    output_frame_dir="viper_output/viper_side_by_side_frames",
):
    """
    Combines each simulation frame with the static decision tree image.
    """

    if os.path.exists(output_frame_dir):
        shutil.rmtree(output_frame_dir)

    os.makedirs(output_frame_dir, exist_ok=True)

    tree_img = Image.open(tree_image_path).convert("RGB")

    frame_files = sorted(
        f for f in os.listdir(sim_frame_dir)
        if f.endswith(".png")
    )

    if len(frame_files) == 0:
        raise RuntimeError(f"No PNG frames found in {sim_frame_dir}")

    for idx, frame_file in enumerate(frame_files):
        sim_path = os.path.join(sim_frame_dir, frame_file)
        sim_img = Image.open(sim_path).convert("RGB")

        sim_w, sim_h = sim_img.size

        # Resize tree to match the simulation frame height.
        tree_aspect = tree_img.width / tree_img.height
        tree_h = sim_h
        tree_w = int(tree_h * tree_aspect)

        resized_tree = tree_img.resize((tree_w, tree_h))

        combined_w = sim_w + tree_w
        combined_h = sim_h

        combined = Image.new("RGB", (combined_w, combined_h), "white")
        combined.paste(sim_img, (0, 0))
        combined.paste(resized_tree, (sim_w, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), "VIPER rollout", fill=(0, 0, 0))
        draw.text((sim_w + 10, 10), "Extracted decision tree", fill=(0, 0, 0))

        out_path = os.path.join(output_frame_dir, f"frame_{idx:04d}.png")
        combined.save(out_path)

    print(f"Saved side-by-side frames to: {output_frame_dir}/")


######################################################################
# Rendering the simulation based on viper actions

def get_action_from_tree(policy, obs):
    """
    Convert environment observation into the shape expected by sklearn's
    DecisionTreeClassifier, then return one discrete action.
    """

    obs = np.asarray(obs, dtype=np.float32)

    # If obs is a single observation with shape (obs_dim,),
    # sklearn expects shape (1, obs_dim).
    if obs.ndim == 1:
        obs_batch = obs.reshape(1, -1)
    else:
        obs_batch = obs

    action = policy.predict(obs_batch)

    # policy.predict returns an array, usually shape (1,)
    if isinstance(action, np.ndarray):
        action = action[0]

    return int(action)


def make_render_video(frame_dir="frames", output_path="viper_final_rollout.mp4", fps=10):
    """
    Create an MP4 from saved PNG frames.
    Requires ffmpeg to be installed.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
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
        print(f"You can still inspect the saved PNG frames in: {frame_dir}/")


def render_viper_rollout(args):
    """
    Load a VIPER decision-tree policy from a joblib file and use it to control
    the agent in one deterministic rollout.
    """

    if os.path.exists(args.frame_dir):
        shutil.rmtree(args.frame_dir)

    os.makedirs(args.frame_dir, exist_ok=True)

    print(f"Loading VIPER policy from: {args.policy_path}")
    policy = joblib.load(args.policy_path)

    print(f"Creating environment: {args.env_name}")
    env = gym.make(args.env_name, args=args)

    # Enable the environment's frame rendering.
    if hasattr(env, "enable_render"):
        env.enable_render(frame_dir=args.frame_dir, clear_old_frames=False)
    else:
        env.render_enabled = True
        env.frame_dir = args.frame_dir

    obs = env.reset()
    done = False

    total_reward = 0.0
    step = 0
    final_info = {}

    print("Running VIPER rollout...")

    while not done:
        action = get_action_from_tree(policy, obs)

        obs, reward, done, info = env.step(action)

        # If your env does not automatically render when render_enabled=True,
        # this fallback forces a frame to be saved.
        if not getattr(env, "render_enabled", False):
            env.render()

        total_reward += float(reward)
        step += 1
        final_info = info

        if args.print_every > 0 and step % args.print_every == 0:
            print(
                f"step={step}, "
                f"reward={float(reward):.3f}, "
                f"total_reward={total_reward:.3f}, "
                f"action={action}, "
                f"status={info.get('status', 'N/A')}, "
                f"num_inspected={info.get('num_inspected', 'N/A')}, "
                f"distance={info.get('distance', 'N/A')}"
            )

        if step >= args.max_steps:
            print(f"Stopping early because max_steps={args.max_steps} was reached.")
            break

    print("\nVIPER rollout complete.")
    print(f"Episode reward: {total_reward:.4f}")
    print(f"Episode steps: {step}")
    print(f"Final info: {final_info}")

    if hasattr(policy, "get_depth") and hasattr(policy, "get_n_leaves"):
        print(f"Tree depth: {policy.get_depth()}")
        print(f"Tree leaves: {policy.get_n_leaves()}")

    tree_image_path = "viper_tree.png"
    side_by_side_dir = "viper_side_by_side_frames"

    save_tree_image(
        policy,
        output_path=tree_image_path,
        max_depth=args.tree_plot_depth,
    )

    make_side_by_side_frames(
        sim_frame_dir=args.frame_dir,
        tree_image_path=tree_image_path,
        output_frame_dir=side_by_side_dir,
    )

    make_render_video(
        frame_dir=side_by_side_dir,
        output_path=args.output_video,
        fps=args.fps,
    )




def main():
    parser = argparse.ArgumentParser(
        description="Render a VIPER joblib decision-tree policy rollout."
    )

    parser.add_argument(
        "--tree-plot-depth",
        type=int,
        default=4,
        help="Maximum tree depth to show in the side-by-side plot. Use a larger value to show more of the tree.",
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="SimpleSingleInspector-v0",
        help="Gym environment name.",
    )

    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to the VIPER .joblib file.",
    )

    parser.add_argument(
        "--frame-dir",
        type=str,
        default="viper_frames",
        help="Directory where PNG frames will be saved.",
    )

    parser.add_argument(
        "--output-video",
        type=str,
        default="viper_final_rollout.mp4",
        help="Output MP4 video path.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Safety limit for rollout steps.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print rollout status every N steps. Use 0 to disable.",
    )

    # These are included so your env can safely receive args=args,
    # similar to your existing code.
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--oracle-path", type=str, default=None)
    parser.add_argument("--rand-ball-start", action="store_true")
    parser.add_argument("--log-prefix", type=str, default="")
    parser.add_argument("--ep-horizon", type=int, default=150)
    parser.add_argument("--n-env", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)

    args = parser.parse_args()

    render_viper_rollout(args)


if __name__ == "__main__":
    main()