## PPO_Vision

Vision-based Proximal Policy Optimization for manipulation tasks in the DISCOVERSE simulator. The policy learns end-to-end from stacked RGB frames to control a robot arm to pick.

### Key Features

- Continuous control using joint-level actions from MuJoCo actuator ranges
- Custom CNN feature extractor integrated into PPO
- Shaped reward with approach/place rewards and smoothness/step penalties
- TensorBoard logging and simple evaluation script

## Folder Structure
- `env.py`: Gymnasium-style environment with image observations, action space, reward, and termination
- `train.py`: PPO training entrypoint with `CNNFeatureExtractor`
- `inference.py`: Model loading and evaluation; includes optional YOLOv10-based utility for image processing
- `README.md`: This file

## Requirements
- Python 3.9+
- PyTorch
- Gymnasium
- MuJoCo (via `mujoco` python package) and DISCOVERSE
- PPO (Stable-Baselines3-compatible fork), version v0.20.0
- Stable-Baselines3 common utilities (callbacks, vec env, monitor)
- Optional (inference utilities): `ultralytics` (YOLOv10), `opencv-python`, `scikit-learn`

Install `sbx` following `https://github.com/araffin/sbx` (use v0.20.0). Make sure DISCOVERSE is installed and its assets are available (objects, scenes, MJCF).

## Observations and Actions
- Observation: stacked RGB frames `(C=3, H=84, W=84, stack=4)` normalized to `[0,1]`
- Action: continuous Box from MuJoCo actuator control ranges (joint targets)

## Reward (high level)
- Approach reward: encourages the end-effector toward the kiwi
- Placement reward: encourages kiwi-to-plate proximity and success
- Step penalty: discourages long episodes
- Action magnitude penalty: encourages smooth control


## Training
Run from `policies/RL/sbx/PPO_Vision`:

```bash
python train.py --total_timesteps 1000000 --batch_size 64 --n_steps 2048 --learning_rate 3e-4 --seed 42 --render
```

Arguments:
- `--render`: enable on-screen rendering
- `--seed`: RNG seed (default 42)
- `--total_timesteps`: total learning steps
- `--batch_size`: mini-batch size
- `--n_steps`: rollout length per update
- `--learning_rate`: optimizer learning rate
- `--log_dir`: optional custom log dir; defaults to `DISCOVERSE_ROOT_DIR/data/PPO_Vision/logs_<timestamp>`
- `--model_path`: optional checkpoint to resume training
- `--log_interval`: logging frequency (default 10)

TensorBoard:

```bash
tensorboard --logdir <DISCOVERSE_ROOT_DIR>/data/PPO_Vision
```

Logged metrics include total and component rewards, policy/value losses, and entropy.

Checkpoints: the final model is saved as `<log_dir>/final_model.zip`.

## Inference / Evaluation

```bash
python inference.py --model_path <path-to-final_model.zip> --episodes 10 --deterministic --render
```

Reports per-episode rewards and aggregate mean/std. The script also shows how to initialize the DISCOVERSE config used in training and includes a `DetectionProcessor` utility with YOLOv10 (optional) for image post-processing.

## Environment Details
Key config used (see `env.py`):
- `cfg.init_key = "pick"`
- Assets: `object/plate_white.ply`, `object/kiwi.ply`, background `scene/tsimf_library_1/point_cloud.ply`
- MJCF: `mjcf/tasks_mmk2/pick_kiwi.xml`
- Camera: RGB camera id `[0]`
- Episode limit: 1000 steps or time limit ~20s


## Tips
- Start with small `n_steps`/`batch_size` to validate end-to-end wiring
- Ensure MuJoCo, assets, and DISCOVERSE paths resolve correctly
- If training is unstable, try lowering `learning_rate` or increasing `n_steps`
- Use deterministic evaluation (`--deterministic`) for reproducibility

## Citation
If you use this in academic work, please cite DISCOVERSE and PPO (Schulman et al.).
