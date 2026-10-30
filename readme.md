# Cooperative Fire Search with Multi-Agent RL (MAPPO)

Teams of agents learn to find a fire inside a building floor plan. Agents see only a small local window plus lidar and thermal rays, share the discovery once one of them finds the fire, and converge on it. The fires are replayed from real CFD (FDS) temperature simulations.

**Question:** does a bigger team find the fire sooner?
**Answer (one 2-agent-trained checkpoint, run with 1 to 4 agents, no retraining):** yes, with diminishing returns beyond the second agent. 

---

## Contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Setup](#setup)
4. [Data layout](#data-layout)
5. [Train](#train)
6. [Evaluate and make GIFs](#evaluate-and-make-gifs)
7. [Results](#results)
8. [Limitations](#limitations)
9. [Configuration reference](#configuration-reference)

---

## How it works

### Environment (`multi_agent_fire_env.py`, `parallel_fire_env.py`)

- Grid world: `0` outside, `1` wall, `2` door, `3` floor, `4` fire.
- `MultiAgentFireEnv` is a Gymnasium environment. `FireParallelEnv` wraps it in the PettingZoo `ParallelEnv` API, so all agents act at the same step and observations, actions and rewards are dicts keyed by agent id.
- **Fire dynamics are real CFD.** Each step reads a frame of the scenario's `T_slices.npy` (temperature, deg C). A cell is on fire when it is above 50% of the scenario's temperature range. One step is 0.4 s, CFD frames are 10 s apart, and the team enters at 60 s.
- **Discovery.** Fire counts as found the moment any agent has an active fire cell inside its own 5x5 window. The fire then freezes and its coordinate becomes visible to every agent. Before that, the true fire location is hidden from all agents.
- **Success.** The episode ends in success when every agent has had fire inside its own window. Otherwise it is truncated at the step cap.
- After discovery, agents that have not reached the fire yet are routed there along a BFS shortest path (`_fire_distance_map`), so the learned policy only has to solve the search phase.

### Observations (`observations.py`)

| Component | Size | Content |
|---|---|---|
| Local view | 100 | 5x5 window, 4 channels: wall, door, fire intensity, teammate |
| Status extras | 9 | own position, door flag, shared fire coordinate (hidden until discovery), total fire, step fraction, blocked-last-step flag |
| Lidar | 16 | normalised distance to nearest wall, range 8 |
| Thermal rays | 16 | hottest reading along each direction up to the first wall (the search signal before discovery) |
| Memory | 7 | visit counts of the 4 neighbours, direction and distance to the nearest unseen cell |
| **Actor input** | **148** | |
| Critic extra | +14 | fixed-size team state (see below) |

**Actions:** `Discrete(5)`: up, down, right, left, stay. Moves into walls, fire or other agents are blocked.

### Rewards

| Term | Value |
|---|---|
| All agents reached fire (terminal) | +10 |
| Fire discovered (once) | +5 |
| First team visit to a cell (novelty) | +0.03 |
| Newly seen cell | +0.002 |
| Distance shaping (progress toward fire, **only with line of sight**) | +0.02 x delta |
| Time cost | -0.01 per step |
| Revisit penalty | -0.01 x visits in last 50 steps (max 5x) |
| Fire-spread penalty | -0.1 x fraction of grid on fire (after discovery) |
| Invalid action / stay | -0.10 / -0.005 |

Distance shaping uses Manhattan distance, which ignores walls. Without the line-of-sight check an agent right behind a wall would be rewarded for pushing into it.

### Learning (`train_mappo.py`, `models/centralized_critic.py`, `callbacks.py`)

MAPPO on Ray RLlib (PPO) with parameter sharing:

- **Actor** (per agent, local observation only): FC 256 -> LSTM 128 -> 5 action logits.
- **Critic** (training only): observation + 14-value team state -> MLP 256 -> 128 -> value.
- **One shared policy** (`shared_policy`) controls every agent.
- **Team state is a fixed 14 values**: fire statistics (step fraction, active fraction, total and max intensity, found flag, all-reached flag, fire centroid y/x, explored fraction) plus team aggregates (fraction of agents at the fire, mean agent position, mean distance to the fire centroid, team size / 8). Aggregates are averages over agents, so the critic input does not depend on agent order or count. This is what lets one checkpoint run with 1 to 4 agents.
- PPO: lr 3e-4, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, value-loss coeff 0.5, batch 4096, minibatch 512, 10 epochs, seed 42.

---

## Repository layout

```
config.py                    experiment and PPO hyper-parameters, run folders
multi_agent_fire_env.py      Gymnasium environment (CFD-driven fire, rewards, discovery)
parallel_fire_env.py         PettingZoo ParallelEnv wrapper
observations.py              local 5x5 view and lidar
fire_dynamics.py             lightweight fire diffusion helper (legacy model)
models/centralized_critic.py LSTM actor + centralised critic (RLlib RecurrentNetwork)
callbacks.py                 per-episode metrics (found / reached / success)
train_mappo.py               MAPPO training with RLlib
zero_shot_eval_marl.py       evaluate a checkpoint on target datasets, no finetuning
generate_gifs_marl.py        GIF rollouts of the trained policy
generate_test_split_gifs.py  GIFs + pass/fail per scenario, logs first-fire step
requirements.txt
```

Outputs created at run time: `checkpoints_<RUN_NAME>/`, `logs/`, `tensorboard/`, `videos/`, `gifs/`.

---

## Setup

Tested versions are pinned in `requirements.txt` (Python 3.11).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Key packages: `ray[rllib]==2.40.0`, `torch==2.4.1`, `pettingzoo==1.25.0`, `gymnasium==1.0.0`, `numpy==1.26.4`.

On Windows PowerShell, set these before running:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:PYTHONUNBUFFERED="1"
```

If a previous run left Ray processes behind, run `ray stop --force` first.

---

## Data layout

The CFD datasets are **not** included in the repository (large). Place each dataset folder in the project root:

```
dataset_multi_new/        Drawing 7
dataset_d8_peft_extended/ Drawing 8
dataset_peft_d9/          Drawing 9
    metadata.json         scenario list, "train_split" and "test_split" ids
    obstacles.npy         building grid shared by all scenarios
    sim_001/
        T_slices.npy      temperature frames, shape (frames, H, W), deg C
        fire_source.npy   ignition location
    sim_002/ ...
```

`mode="train"` reads `metadata["train_split"]`, `mode="test"` reads `metadata["test_split"]`. Two further buildings (`dataset_d10`, `dataset_d13`) are reserved for zero-shot evaluation on unseen layouts and are deliberately excluded from training.

Scenarios used here: 27 total across 3 buildings (17 train, 10 test).

---

## Train

```bash
python train_mappo.py --iterations 200 2>&1 | tee train_log.txt
python train_mappo.py --iterations 200 --resume checkpoints_reach_mem/checkpoint_000100
```

Checkpoints are saved every 20 iterations to `checkpoints_<RUN_NAME>/checkpoint_XXXXXX`. Change `RUN_NAME` (or set the `RUN_NAME` environment variable) to keep runs separate.

**Reproducing the reported checkpoint:** it was trained with **2 agents only**. `TRAIN_AGENT_RANGE` in `config.py` controls the team size drawn per training episode; set it to `(2, 2)` before training to match. The shipped default is `(1, 4)`.

---

## Evaluate and make GIFs

All scripts load the newest checkpoint of the current run by default, or take a checkpoint folder as the first argument. `--agents N` sets the team size at evaluation time (the same weights run with any N).

```bash
# Pass/fail per scenario on train and test splits, GIF per episode, first-fire CSV
python generate_test_split_gifs.py --agents 1
python generate_test_split_gifs.py --agents 2
python generate_test_split_gifs.py --agents 3
python generate_test_split_gifs.py --agents 4

# Zero-shot success rate on the target datasets
python zero_shot_eval_marl.py --agents 4

# GIF rollouts only
python generate_gifs_marl.py --agents 2
```

`generate_test_split_gifs.py` writes GIFs and a per-agent-count `log.txt` under `gifs/`, and `first_found_a{N}.csv` with, for each scenario, the step at which any agent first sees the fire. Evaluation cap is 600 steps.

---

## Results

One checkpoint (iteration 200), trained with 2 agents, run with 1 to 4 agents on 27 scenarios (3 buildings, train and test splits). Greedy actions, one episode per scenario.

| Agents | Scenarios solved | Mean steps to first fire* | Median steps to first fire** |
|---|---|---|---|
| 1 | 23 / 27 | 108 | 88 |
| 2 | 27 / 27 | 69 | 52 |
| 3 | 27 / 27 | 63 | 41 |
| 4 | 27 / 27 | 62 | 30 |

\* Mean over the 22 scenarios where every team size had a readable first-fire step.
\*\* Median over all 27, with never-found counted as 600.

- Mean steps to first fire fall about 43% from 1 to 4 agents (median about 51% on the 22-scenario set).
- Going from 1 to 2 agents is faster in 13 of 22 scenarios and never slower. From 2 to 3 only 4 improve; from 3 to 4, 5 improve and 2 get slower.
- Training: success rises from 0% to about 96% and mean episode length falls from the step cap to about 120 over 200 iterations.

Time for the **whole team** to arrive stays roughly flat (about 103 to 116 steps), because the episode ends when the last agent reaches the fire.

---

## Limitations

- One deterministic episode per scenario and team size: no error bars or significance tests.
- First-fire steps in the table above were read from replay GIFs (first frame with fire within 2 cells of an agent), not logged. Use the `first_found_a{N}.csv` from the current `generate_test_split_gifs.py` for exact values.
- No random-action baseline yet.
- 17 of the 27 scenarios are training scenarios; held-out numbers should be read from the test split only.
- Grid world with replayed CFD fires. Smoke and visibility are not modelled, and nothing here is validated on a physical system.
- Four agents are slower than two on two hard Drawing 8 test floors.

---

## Configuration reference

Edit `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `NUM_AGENTS` | 2 | evaluation-time team size |
| `TRAIN_AGENT_RANGE` | `(1, 4)` | team size drawn per training episode |
| `NUM_WORKERS` | 2 | RLlib rollout workers |
| `TRAIN_BATCH_SIZE` / `SGD_MINIBATCH_SIZE` | 4096 / 512 | PPO batch sizes |
| `LR`, `GAMMA`, `LAMBDA` | 3e-4, 0.99, 0.95 | optimisation |
| `CLIP_PARAM`, `ENTROPY_COEFF`, `VF_LOSS_COEFF` | 0.2, 0.01, 0.5 | PPO loss terms |
| `CHECKPOINT_FREQ` | 20 | iterations between checkpoints |
| `RUN_NAME` | `reach_mem` | name of the checkpoint folder |
| `SEED` | 42 | random seed |






<!-- $env:OPENBLAS_NUM_THREADS="1"
D:\fire_search_project\.venv\Scripts\Activate.ps1


D:\fire_search_project\.venv\Scripts\Activate.ps1
cd D:\Downloads\mappo_fire\mappo_fire
ray stop --force
$env:OPENBLAS_NUM_THREADS="1"
$env:PYTHONUNBUFFERED="1"
python train_mappo.py --iterations 200 2>&1 | Tee-Object -FilePath train_log_1.txt

python generate_test_split_gifs.py --agents 3
python generate_test_split_gifs.py --agents 4
python zero_shot_eval_marl.py --agents 4 -->