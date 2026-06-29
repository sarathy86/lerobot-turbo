# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation (development)
```bash
pip install -e ".[dev,test]"
pre-commit install
```

### Linting & formatting
```bash
pre-commit run --all-files   # run all checks (ruff format + lint, typos, etc.)
ruff check src/ tests/       # lint only
ruff format src/ tests/      # format only
```

### Running tests
```bash
# Prerequisites: git-lfs must be installed and pulled
git lfs install && git lfs pull

pytest -sv ./tests                         # full suite
pytest -sv tests/test_specific_feature.py  # single file
pytest -sv tests/datasets/test_datasets.py # specific subdirectory
```

Test device is auto-selected (CUDA if available), or override:
```bash
LEROBOT_TEST_DEVICE=cpu pytest -sv ./tests
```

### End-to-end training/eval (via Makefile)
```bash
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
make test-end-to-end DEVICE=cpu   # runs all E2E tests
```

### CLI entry points (after installation)
```bash
lerobot-train   --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
lerobot-eval    --policy.path=<checkpoint_dir>
lerobot-record  # data collection
lerobot-teleoperate
lerobot-calibrate
lerobot-info    # show environment info
```

## Architecture Overview

### Config system (`src/lerobot/configs/`)
All configs use **draccus** (dataclass-based CLI parser). The key entry points are:
- `TrainPipelineConfig` (`configs/train.py`): top-level training config composing `DatasetConfig`, `EnvConfig`, `PreTrainedConfig` (policy), optimizer, and scheduler.
- `EvalPipelineConfig` (`configs/eval.py`): evaluation counterpart.
- `PreTrainedConfig` (`configs/policies.py`): base class for all policy configs; uses `draccus.ChoiceRegistry` so `--policy.type=act` auto-selects the right subclass.

CLI flags map directly to nested dataclass fields (`--policy.dim_model=64`, `--dataset.repo_id=...`). Configs can also be loaded from a saved JSON (`--config_path=.../train_config.json`).

### Policies (`src/lerobot/policies/`)
Each policy lives in its own subdirectory (e.g., `act/`, `diffusion/`, `smolvla/`, `tdmpc/`). All policies:
- Subclass `PreTrainedPolicy` (`policies/pretrained.py`), which extends `nn.Module` + `HubMixin` (push/pull from HF Hub).
- Have a paired `PreTrainedConfig` subclass registered via `draccus.ChoiceRegistry`.
- Implement `select_action(batch) -> Tensor` and `forward(batch) -> loss dict`.

Available policy types: `act`, `diffusion`, `smolvla`, `tdmpc`, `vqbet`, `pi0`, `pi05`, `pi0_fast`, `groot`, `xvla`, `sarm`, `wall_x`, `rtc`, `sac`.

### Datasets (`src/lerobot/datasets/`)
`LeRobotDataset` is the central class. Data format: Parquet files for state/action + MP4 or image files for vision, stored locally or streamed from HF Hub. Datasets track episodes, tasks, and optional subtasks. Stats are precomputed per feature.

### Robots (`src/lerobot/robots/`)
`Robot` (`robots/robot.py`) is the abstract base. Each hardware type (SO-100, LeKiwi, Koch, Reachy2, Unitree G1, etc.) lives in its own subdirectory with a `RobotConfig` dataclass. Robots expose `connect()`, `disconnect()`, `get_observation() -> RobotObservation`, and `send_action(action: RobotAction)`.

### Processor pipeline (`src/lerobot/processor/`)
A composable `DataProcessorPipeline` transforms data between robot/env and policy. `ProcessorStep` subclasses handle normalization, observation renaming, device transfer, action delta computation, tokenization, etc. The pipeline is serializable and can be pushed to HF Hub alongside the policy.

### RL support (`src/lerobot/rl/`)
Contains actor/learner processes for online RL (SAC-based HIL-SERL), replay buffers, and Gymnasium environment wrappers (`gym_manipulator.py`). The actor and learner run as separate processes communicating via queues/gRPC.

### Async inference (`src/lerobot/async_inference/`)
gRPC-based client/server split: `policy_server.py` runs inference, `robot_client.py` runs control. Enables decoupled inference on a remote GPU while the robot control loop runs locally.

### Transport (`src/lerobot/transport/`)
Protobuf/gRPC definitions (`services.proto`, generated `*_pb2.py`) used by async inference and RL actor-learner communication.

## Key Conventions

- **Line length:** 110 characters (ruff enforced).
- **Python minimum:** 3.10; use `|` union syntax, match statements where appropriate.
- **Imports:** isort enforced; `lerobot` is `known-first-party`.
- **Docstrings:** Google style when present.
- **Mypy:** Gradually being enabled module by module; `configs.*`, `optim.*`, `model.*`, `cameras.*`, `envs.*`, `transport.*` have strict checks. New modules should aim to pass mypy.
- **Optional extras:** Hardware and policy extras are separate (e.g., `pip install -e ".[act,aloha]"`). Check `pyproject.toml` for the full list before importing hardware-specific packages.
- **Hub integration:** `HubMixin` (from `lerobot.utils.hub`) adds `push_to_hub` / `from_pretrained` to configs and policies.
- **Artifacts in tests:** Binary test artifacts live in `tests/artifacts/` and are tracked via git-lfs.
