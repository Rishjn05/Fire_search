import os
from pathlib import Path

# Experiment configuration
NUM_AGENTS = 2          # eval-time team size (training uses the range below)
TRAIN_AGENT_RANGE = (1, 4)  # team size drawn uniformly from this range every training episode
FRAMEWORK = "torch"
NUM_WORKERS = 2
NUM_ENVS_PER_WORKER = 1
ROLLOUT_FRAGMENT_LENGTH = 512
TRAIN_BATCH_SIZE = 4096
SGD_MINIBATCH_SIZE = 512
NUM_SGD_ITER = 10
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_PARAM = 0.2
ENTROPY_COEFF = 0.01
VF_LOSS_COEFF = 0.5
MAX_EPISODE_STEPS = 500
CHECKPOINT_FREQ = 20
EVALUATION_INTERVAL = 10
EVALUATION_EPISODES = 20
SEED = 42

# Directories
ROOT = Path(__file__).resolve().parent
# Each run saves to its own folder, so earlier checkpoints are never touched.
# Change RUN_NAME (or set env var RUN_NAME) to start a new, separate set.
RUN_NAME = os.environ.get("RUN_NAME", "reach_mem")
CHECKPOINT_DIR = ROOT / f"checkpoints_{RUN_NAME}"
LOG_DIR = ROOT / "logs"
TENSORBOARD_DIR = ROOT / "tensorboard"
VIDEOS_DIR = ROOT / "videos"

for d in (CHECKPOINT_DIR, LOG_DIR, TENSORBOARD_DIR, VIDEOS_DIR):
    os.makedirs(d, exist_ok=True)

# Useful helper
def checkpoint_path(iteration: int) -> str:
    return str(CHECKPOINT_DIR / f"checkpoint_{iteration:06d}")