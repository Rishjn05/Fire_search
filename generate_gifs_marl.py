"""
Generates GIFs of the trained MAPPO policy (checkpoints/) running on
train/test-split scenarios -- i.e. the buildings the policy actually
trained on (dataset_multi_new, dataset_d8_peft_extended, dataset_peft_d9),
split into a train_split example (seen scenario) and a test_split example
(held-out scenario, same building) per dataset. This checks in-distribution
performance and train-vs-test generalization on already-seen buildings.

For unseen-building zero-shot GIFs (dataset_d10, dataset_d13), use
generate_zero_shot_gifs.py instead -- this file no longer covers those.

Usage (from fire_search_project root, with your venv active):
    python generate_gifs_marl.py
"""

import json
import os
import imageio
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from parallel_fire_env import FireParallelEnv
from models.centralized_critic import CentralizedCriticModel

# ---- config ----
import argparse as _argparse
import glob as _glob


from config import CHECKPOINT_DIR as _CKPT_DIR


def _latest_checkpoint(root=str(_CKPT_DIR)):
    """Newest per-iteration folder (checkpoints/checkpoint_000200, ...) that
    train_mappo.py writes; falls back to the old single-folder layout."""
    subs = sorted(d for d in _glob.glob(os.path.join(root, "checkpoint_*")) if os.path.isdir(d))
    return subs[-1] if subs else root


_parser = _argparse.ArgumentParser()
_parser.add_argument("checkpoint", nargs="?", default=None,
                     help="checkpoint folder (default: newest under the current run's checkpoint dir, see config.RUN_NAME)")
_parser.add_argument("--agents", type=int, default=2,
                     help="number of agents at evaluation time (no retraining needed)")
_args = _parser.parse_args()
CHECKPOINT_PATH = _args.checkpoint or _latest_checkpoint()

NUM_AGENTS = _args.agents
MAX_STEPS = 600
POLICY_ID = "shared_policy"
OUTPUT_DIR = "gifs"
FPS = 15
# Must match train_mappo.py's LIDAR_RAYS -- determines observation vector
# size, which is baked into the checkpoint's policy network input layer.
LIDAR_RAYS = 16

# Same three buildings train_mappo.py trains/evaluates on.
TRAIN_DATASETS = ["dataset_multi_new", "dataset_d8_peft_extended", "dataset_peft_d9"]


def get_split_sim(dataset_dir, split_key, index=0):
    """Read metadata.json and return the sim id at `index` within the
    given split ("train_split" or "test_split"), or None if unavailable."""
    meta_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    split = meta.get(split_key)
    if not split or index >= len(split):
        return None
    return split[index]


def build_gif_jobs():
    """One train_split + one test_split job per training dataset, so each
    GIF filename records exactly which dataset/split/scenario it is."""
    jobs = []
    for dataset_dir in TRAIN_DATASETS:
        for split_key, split_label in [("train_split", "TRAIN"), ("test_split", "TEST")]:
            sim_id = get_split_sim(dataset_dir, split_key)
            if sim_id is None:
                print(f"[skip] {dataset_dir}: no {split_key} entry found")
                continue
            filename = f"{dataset_dir}_{sim_id}_{split_label}.gif"
            jobs.append((dataset_dir, sim_id, filename))
    return jobs


def run_episode_and_record(policy, dataset_dir, scenario_id):
    env = FireParallelEnv(
        dataset_dir=dataset_dir,
        mode="all",
        num_agents=NUM_AGENTS,
        training=False,
        max_steps=MAX_STEPS,
        lidar_rays=LIDAR_RAYS,
    )
    obs, info = env.reset(options={"scenario_id": scenario_id})
    state = {aid: policy.get_initial_state() for aid in obs.keys()}

    frames = [env.render()]
    done = {"__all__": False}
    steps = 0
    found_fire = False

    while not done.get("__all__", False) and steps < MAX_STEPS:
        actions = {}
        for agent_id, agent_obs in obs.items():
            action, state[agent_id], _ = policy.compute_single_action(
                agent_obs, state=state[agent_id], explore=False
            )
            actions[agent_id] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)
        done["__all__"] = all(terminations.values()) or all(truncations.values())
        frames.append(env.render())

        for info_val in infos.values():
            if isinstance(info_val, dict) and info_val.get("success"):
                found_fire = True
        steps += 1

    env.close()
    return frames, found_fire, steps


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

    def _creator(_env_config):
        return ParallelPettingZooEnv(
            FireParallelEnv(dataset_dir="dataset_multi_new", mode="train",
                             num_agents=NUM_AGENTS, training=True, lidar_rays=LIDAR_RAYS)
        )
    register_env("fire_parallel_env", _creator)

    def _eval_creator(_env_config):
        return ParallelPettingZooEnv(
            FireParallelEnv(dataset_dir=TRAIN_DATASETS, mode="test",
                             num_agents=NUM_AGENTS, training=False, lidar_rays=LIDAR_RAYS)
        )
    register_env("fire_parallel_env_eval", _eval_creator)

    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
    algo = Algorithm.from_checkpoint(os.path.abspath(CHECKPOINT_PATH))
    policy = algo.get_policy(POLICY_ID)

    gif_jobs = build_gif_jobs()
    for dataset_dir, sim_id, filename in gif_jobs:
        if not os.path.isdir(dataset_dir):
            print(f"[skip] {dataset_dir} not found")
            continue

        print(f"Running {dataset_dir}/{sim_id} ...")
        frames, found_fire, steps = run_episode_and_record(policy, dataset_dir, sim_id)

        out_path = os.path.join(OUTPUT_DIR, filename)
        imageio.mimsave(out_path, frames, fps=FPS)
        status = "SUCCESS" if found_fire else "did not succeed"
        print(f"  -> saved {out_path} ({steps} steps, {status})")

    algo.stop()
    ray.shutdown()
    print(f"\nDone. GIFs saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()