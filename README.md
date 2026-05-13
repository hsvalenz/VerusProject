# Viper Spacecraft Inspection Project

This repository is adapted from the following repositories:

- https://github.com/act3-ace/safe-autonomy-simulation.git
- https://github.com/Safe-RL-Team/viper-verifiable-rl-impl.git

Read the accompanying VIPER blog post [here](https://safe-rl-team.github.io/viper-verifiable-reinforcement-learning/).

**V**erifiability via **I**terative **P**olicy **E**xt**R**action (VIPER), 2019  
Paper: https://arxiv.org/abs/1805.08328

VIPER distills a deep reinforcement learning oracle policy into an interpretable decision tree policy. In this project, PPO is used as the oracle policy, and VIPER is used to extract a decision tree for simplified spacecraft inspection environments.

---

## Project Overview

This repository implements and tests VIPER on spacecraft inspection environments:

- `SimpleSingleInspector-v0`
- `FuelSingleInspector-v0`

The agent represents a deputy spacecraft inspecting an RSO/chief target. The deputy moves in 3D space using discrete thrust actions. The task is to inspect points on the target surface while avoiding unsafe behavior. The fuel-aware environment also requires the agent to consider fuel usage.

The environment uses simplified damped point-mass dynamics instead of full CWH/HCW orbital dynamics. This reduces training complexity while preserving the main decision-making problem: navigating around a target, inspecting surface points, and selecting meaningful thrust actions.

---

## Environments

### `SimpleSingleInspector-v0`

This environment includes:

- 3D deputy position and velocity
- Fixed RSO/chief target at the origin
- Discrete surface inspection points
- Desired viewing point for reward shaping
- 27 discrete thrust actions
- Rewards for newly inspected points
- Penalties for speed, unsafe approach, crash, timeout, and out-of-bounds behavior

### `FuelSingleInspector-v0`

This environment extends `SimpleSingleInspector-v0` with fuel awareness:

- Adds `fuel_remaining_fraction` to the observation space
- Reduces fuel based on the norm of the thrust command
- Penalizes fuel usage in the reward function
- Adds an out-of-fuel termination condition
- Logs fuel metrics to TensorBoard

---

## Observation and Action Space

For `FuelSingleInspector-v0`, the observation vector is:

```text
[
    x, y, z,
    vx, vy, vz,
    sun_angle,
    num_inspected,
    target_dir_x, target_dir_y, target_dir_z,
    fuel_remaining_fraction
]

```
The action space is discrete with 27 actions:
- 3 thrust options per axis: {-1, 0, 1}
- 3^3 = 27 total actions

For viper, the input features are the oberservation values and the output classes are discrete action IDs (ex. a4 = [-1,-1,0])

## Usage

First, create a python virtual environment based on `python3.10`. Once activated, pip install all requirements found in `requirements.txt:
```
pip install -r requirements.txt
```

The project is run through `main.py`, to see all available options:
```
python3 main.py --help
```

### Training and testing `SimpleSingleInspector-v0`

Train Oracle:
```
python3 main.py train-oracle \
  --env-name SimpleSingleInspector-v0 \
  --n-env 32 \
  --total-timesteps 500000 \
  --oracle-path oracle_SimpleSingleInspector-v0.zip
```

to render a final rollout after training append `--render` to train-oracle.

Test Oracle:
```
python3 main.py test-oracle \
  --env-name SimpleSingleInspector-v0 \
  --oracle-path oracle_SimpleSingleInspector-v0.zip
```

Train VIPER:
```
python3 main.py train-viper \
  --env-name SimpleSingleInspector-v0 \
  --n-env 32 \
  --oracle-path oracle_SimpleSingleInspector-v0.zip \
  --max-depth 12 \
  --max-leaves 32 \
  --n-iter 10 \
  --total-timesteps 500000
```

Test VIPER:
```
python3 main.py test-viper \
  --env-name SimpleSingleInspector-v0 \
  --oracle-path oracle_SimpleSingleInspector-v0.zip \
  --max-depth 12 \
  --max-leaves 32
```

Same commands from above apply to FuelSingleInspector-v0 as well.

To obtain training logs, use `Tensorboard` training logs, which are saved in `log/`. 
Start Tensorboard with:
```
tensorboard --logdir ./log --reload_interval 5
```

Finally, to render the VIPER rollout, use `render_viper.py` to load a saved VIPER decision tree.

Render `SimpleSingleInspector-v0`:
```
python3 render_viper.py \
  --env-name SimpleSingleInspector-v0 \
  --policy-path ./log/viper_SimpleSingleInspector-v0_32_12.joblib \
  --output-video simple_viper_tree_rollout.mp4 \
  --tree-plot-depth 4
```

Render `FuelSingleInpsector-v0`
```
python3 render_viper.py \
  --env-name FuelSingleInspector-v0 \
  --policy-path ./log/viper_FuelSingleInspector-v0_32_12.joblib \
  --output-video fuel_viper_tree_rollout.mp4 \
  --tree-plot-depth 4
```

`render_viper.py` will generate the following:
- Simulation frames 
- Decision Tree image
- Side-by-side rollout and tree video