"""
Generates one GIF per scenario -- both train_split and test_split -- for
the given datasets, using the current checkpoint, and prints a pass/fail
(success) summary per scenario.

This answers a specific question aggregate eval numbers can't: which exact
scenario(s) are failing when [eval, test_split] success_mean dips (e.g.
iterations 80/100/280/310/320 in the training log), and what the failure
actually looks like. Train-split GIFs are included too, as a sanity check
that the policy behaves as expected on scenarios it was actually trained on.

Usage (from fire_search_project root, with your venv active):
    python generate_test_split_gifs.py

Requires: PIL (pillow) for GIF encoding -- likely already installed since
you already have generate_gifs_marl.py in the project.
"""

import json
import os

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from PIL import Image

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
                # doesn't require retraining -- same pattern as
                # zero_shot_eval_marl.py's NUM_AGENTS = 2
MAX_STEPS = 600
POLICY_ID = "shared_policy"
OUTPUT_DIR = "gifs"
# Must match train_mappo.py's LIDAR_RAYS -- determines observation vector
# size, which is baked into the checkpoint's policy network input layer.
LIDAR_RAYS = 16

# d7 = dataset_multi_new, d8 = dataset_d8_peft_extended, d9 = dataset_peft_d9
TARGET_DATASETS = [
    "dataset_multi_new",
    "dataset_d8_peft_extended",
    "dataset_peft_d9",
]

# Both splits get rendered. Set to just ["test"] if you only want failures
# to inspect, or just ["train"] to sanity-check trained-on scenarios.
SPLITS = ["train", "test"]


def get_split_ids(dataset_dir, split):
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    key = "train_split" if split == "train" else "test_split"
    return meta.get(key, [])


def run_episode_and_record(policy, env, sim_id, max_steps=MAX_STEPS):
    """Run one episode pinned to sim_id, return (success, frames)."""
    obs, info = env.reset(options={"scenario_id": sim_id})
    done = {"__all__": False}
    state = {aid: policy.get_initial_state() for aid in obs.keys()}

    frames = [env.render()]
    success = False
    steps = 0
    while not done.get("__all__", False) and steps < max_steps:
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
                success = True
        steps += 1

    return success, frames


def save_gif(frames, path, fps=10):
    images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

    def _creator(_env_config):
        return ParallelPettingZooEnv(
            FireParallelEnv(
                dataset_dir="dataset_multi_new", mode="train",
                num_agents=NUM_AGENTS, training=True, lidar_rays=LIDAR_RAYS,
            )
        )
    register_env("fire_parallel_env", _creator)

    # The checkpoint's restored config includes the .evaluation(...) block
    # from train_mappo.py, which points at "fire_parallel_env_eval" --
    # Algorithm.from_checkpoint() spins up eval env runners using that
    # name, so it must be registered too or restore fails with an
    # unregistered-env error (this is what caused the IndexError above:
    # the eval env runner died at construction, leaving zero eval workers).
    def _eval_creator(_env_config):
        return ParallelPettingZooEnv(
            FireParallelEnv(
                dataset_dir="dataset_multi_new", mode="test",
                num_agents=NUM_AGENTS, training=False, lidar_rays=LIDAR_RAYS,
            )
        )
    register_env("fire_parallel_env_eval", _eval_creator)

    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
    algo = Algorithm.from_checkpoint(os.path.abspath(CHECKPOINT_PATH))
    policy = algo.get_policy(POLICY_ID)

    summary = {}
    for dataset_dir in TARGET_DATASETS:
        if not os.path.isdir(dataset_dir):
            print(f"[skip] {dataset_dir} not found")
            continue

        for split in SPLITS:
            split_ids = get_split_ids(dataset_dir, split)
            if not split_ids:
                print(f"[warn] {dataset_dir} has no {split}_split entries, skipping")
                continue

            print(f"\n--- {dataset_dir} [{split}] (ids: {split_ids}) ---")

            # mode=split controls which pool MultiAgentFireEnv reads from
            # metadata.json (train_split vs test_split) -- a fresh env is
            # needed per split since that's decided at construction time.
            env = FireParallelEnv(
                dataset_dir=dataset_dir, mode=split,
                num_agents=NUM_AGENTS, training=False, max_steps=MAX_STEPS,
                lidar_rays=LIDAR_RAYS,
            )

            for sim_id in split_ids:
                success, frames = run_episode_and_record(policy, env, sim_id)
                status = "SUCCESS" if success else "FAILED "
                print(f"  {sim_id}: {status}  ({len(frames)} steps)")
                summary[f"{dataset_dir}/{split}/{sim_id}"] = success

                gif_name = f"{dataset_dir}_{split}_{sim_id}_a{NUM_AGENTS}_{'pass' if success else 'FAIL'}.gif"
                save_gif(frames, os.path.join(OUTPUT_DIR, gif_name))

            env.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for v in summary.values() if v)
    n_total = len(summary)
    for key, ok in summary.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {key}")
    print(f"\n{n_pass}/{n_total} scenarios (train + test) succeeded.")
    print(f"GIFs saved to ./{OUTPUT_DIR}/ -- filenames ending in _FAIL are the ones to look at first.")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()