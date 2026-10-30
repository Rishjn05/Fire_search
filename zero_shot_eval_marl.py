"""
Zero-shot evaluation of the trained MAPPO checkpoint (RLlib, shared policy
across N agents) on new buildings -- NO finetuning, checkpoint weights
untouched.

Usage (from fire_search_project root, with your venv active):
    python zero_shot_eval_marl.py

Edit CHECKPOINT_PATH / TARGET_DATASETS / N_EPISODES_PER_SIM below if needed.
"""

import json
import os
import numpy as np
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

TARGET_DATASETS = [
    "dataset_d8_peft_extended",
    "dataset_peft_d9",
    "dataset_d10",
    "dataset_d13",
]

N_EPISODES_PER_SIM = 10
MAX_STEPS = 600
POLICY_ID = "shared_policy"

# Must match train_mappo.py's LIDAR_RAYS exactly -- this determines the
# observation vector size, which is baked into the checkpoint's policy
# network input layer. A mismatch here doesn't just give wrong readings,
# it fails to restore the checkpoint at all (shape mismatch).
LIDAR_RAYS = 16

# Datasets train_mappo.py trains/evaluates on -- needed here only because
# newer checkpoints save an evaluation_config pointing at the
# "fire_parallel_env_eval" env name, and Algorithm.from_checkpoint()
# rebuilds the FULL algorithm (training env runners AND eval env runners)
# from that saved config. If "fire_parallel_env_eval" isn't registered in
# THIS process too, checkpoint loading crashes with EnvError even though
# this script never actually uses those eval workers for anything -- it
# only needs the policy object. The content of this creator doesn't matter
# for our purposes, it just has to resolve to a valid, matching env.
TRAIN_DATASETS = ["dataset_multi_new", "dataset_d8_peft_extended", "dataset_peft_d9"]


def get_all_sim_ids(dataset_dir):
    with open(os.path.join(dataset_dir, "metadata.json")) as f:
        meta = json.load(f)
    return [s["id"] for s in meta["simulations"]]


def evaluate_on_sim(policy, env, sim_id, n_episodes=N_EPISODES_PER_SIM):
    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(options={"scenario_id": sim_id})
        done = {"__all__": False}
        state = {aid: policy.get_initial_state() for aid in obs.keys()}

        episode_success = False
        steps = 0
        while not done.get("__all__", False) and steps < MAX_STEPS:
            actions = {}
            for agent_id, agent_obs in obs.items():
                action, state[agent_id], _ = policy.compute_single_action(
                    agent_obs, state=state[agent_id], explore=False
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            done["__all__"] = all(terminations.values()) or all(truncations.values())

            for info_val in infos.values():
                if isinstance(info_val, dict) and info_val.get("success"):
                    episode_success = True
            steps += 1

        if episode_success:
            successes += 1

    return (successes / n_episodes) * 100


def evaluate_on_dataset(policy, dataset_dir, n_episodes_per_sim=N_EPISODES_PER_SIM):
    sim_ids = get_all_sim_ids(dataset_dir)
    env = FireParallelEnv(
        dataset_dir=dataset_dir,
        mode="all",
        num_agents=NUM_AGENTS,
        training=False,
        max_steps=MAX_STEPS,
        lidar_rays=LIDAR_RAYS,
    )

    per_sim_results = {}
    for sim_id in sim_ids:
        rate = evaluate_on_sim(policy, env, sim_id, n_episodes_per_sim)
        per_sim_results[sim_id] = rate
        print(f"  {sim_id}: {rate:.1f}%")

    env.close()
    overall = sum(per_sim_results.values()) / len(per_sim_results) if per_sim_results else 0.0
    return overall, per_sim_results


def main():
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

    print("\n" + "=" * 60)
    print("ZERO-SHOT MARL EVALUATION (no finetuning)")
    print("=" * 60)

    all_results = {}
    for dataset_dir in TARGET_DATASETS:
        if not os.path.isdir(dataset_dir):
            print(f"\n[skip] {dataset_dir} not found")
            continue

        print(f"\n--- {dataset_dir} ---")
        overall, per_sim = evaluate_on_dataset(policy, dataset_dir)
        print(f"  OVERALL: {overall:.1f}%")
        all_results[dataset_dir] = {"overall_success_rate": overall, "per_sim": per_sim}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_dir, res in all_results.items():
        print(f"  {dataset_dir}: {res['overall_success_rate']:.1f}% zero-shot success")

    with open("zero_shot_eval_marl_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results to zero_shot_eval_marl_results.json")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()