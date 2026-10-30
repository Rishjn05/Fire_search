from typing import Dict, Any
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class FireCallbacks(DefaultCallbacks):
    """Custom callbacks collecting episode metrics for MAPPO experiments.

    This callback writes relevant metrics to `episode.custom_metrics` so
    they are available in TensorBoard / result dicts.

    The centralized-critic state is NOT injected here any more: the env
    appends it to every agent's observation (see MultiAgentFireEnv
    `append_global_state`), so it is aligned with the obs by construction.
    """

    @staticmethod
    def _first_env(envs):
        """Pull a single env instance out of whatever get_unwrapped() returned.

        BUG FIXED: previous code assumed `envs[0]` always works, i.e. that
        get_unwrapped() always returns a list/tuple. Depending on RLlib
        version, it can instead return a dict keyed by sub-env index (e.g.
        {0: env, 1: env, ...}), in which case envs[0] on a dict without an
        integer key 0 raises KeyError -- which was then silently swallowed
        by the bare except below, turning a real bug into a quiet None.
        This handles list/tuple, dict, and bare-instance return shapes.
        """
        if isinstance(envs, (list, tuple)):
            return envs[0] if envs else None
        if isinstance(envs, dict):
            if not envs:
                return None
            # Prefer key 0 if present, else just take any single value.
            if 0 in envs:
                return envs[0]
            return next(iter(envs.values()))
        # Not a container -- assume get_unwrapped() returned a bare env.
        return envs

    @staticmethod
    def _get_par_env(worker=None, base_env=None, debug=False):
        """Resolve down to the FireParallelEnv/MultiAgentFireEnv layer shared
        by on_episode_end, on_episode_step, and on_postprocess_trajectory.

        BUG FIXED (confirmed via debug logging): worker.env.get_unwrapped()
        was assumed to exist because worker.env was assumed to be a BaseEnv
        wrapper. In this setup it's actually already the raw
        ParallelPettingZooEnv (a MultiAgentEnv) -- 'ParallelPettingZooEnv'
        object has no attribute 'get_unwrapped' at all, confirmed by the
        AttributeError this used to swallow silently. Fixed by checking
        worker.env directly for a .par_env attribute first (no unwrap
        call needed), and only falling back to get_unwrapped() paths for
        RLlib configurations where env really is BaseEnv-wrapped.
        """
        env = None

        # Primary path for this project's setup: worker.env IS the
        # ParallelPettingZooEnv instance already -- no get_unwrapped() call
        # exists or is needed.
        if worker is not None and hasattr(worker, "env") and worker.env is not None:
            candidate = worker.env
            if hasattr(candidate, "par_env"):
                env = candidate
                if debug:
                    print(f"[FireCallbacks DEBUG] worker.env used directly: {type(candidate)}")
            elif hasattr(candidate, "get_unwrapped"):
                try:
                    envs = candidate.get_unwrapped()
                    env = FireCallbacks._first_env(envs)
                except Exception as exc:
                    if debug:
                        print(f"[FireCallbacks DEBUG] worker.env.get_unwrapped() failed: {exc!r}")
                    env = None

        # Fallback: base_env, either directly (if it already has .par_env)
        # or via get_unwrapped() for RLlib configs where base_env really is
        # BaseEnv-wrapped.
        if env is None and base_env is not None:
            if hasattr(base_env, "par_env"):
                env = base_env
                if debug:
                    print(f"[FireCallbacks DEBUG] base_env used directly: {type(base_env)}")
            elif hasattr(base_env, "get_unwrapped"):
                try:
                    envs = base_env.get_unwrapped()
                    env = FireCallbacks._first_env(envs)
                except Exception as exc:
                    if debug:
                        print(f"[FireCallbacks DEBUG] base_env.get_unwrapped() failed: {exc!r}")
                    env = None

        if env is not None and hasattr(env, "par_env"):
            env = env.par_env
        if debug:
            print(f"[FireCallbacks DEBUG] _get_par_env resolved to: {type(env)}")
        return env

    def on_episode_end(self, *, worker=None, base_env=None, policies=None, episode=None, **kwargs):
        # episode: Episode object
        info = episode.last_info_for("agent_0") or {}

        # Total reward and length
        total_reward = episode.total_reward
        length = episode.length

        # fire_found: any agent found the fire. success == all agents
        # reached the fire (terminal).
        fire_found = False
        all_agents_reached_fire = False
        success = False
        cells_visited = 0
        positions = None
        episode_max_temp = 0.0

        # Try to extract from last info
        if isinstance(info, dict):
            fire_found = bool(info.get("fire_found", False))
            all_agents_reached_fire = bool(info.get("all_agents_reached_fire", False))
            success = bool(info.get("success", False))
            cells_visited = int(info.get("cells_visited", 0))
            positions = info.get("positions", None)
            episode_max_temp = float(info.get("episode_max_temp", 0.0))

        # Compute coverage if possible.
        # Uses the shared _get_par_env helper (worker.env.get_unwrapped()
        # first, falling back to base_env.get_unwrapped(), then unwrapping
        # ParallelPettingZooEnv -> FireParallelEnv), matching how
        # on_episode_step/on_postprocess_trajectory resolve the env.
        # Confirmed working via debug logging (coverage_mean nonzero,
        # ~0.09 across evals) -- debug prints removed now that it's
        # verified; logic unchanged.
        coverage = 0.0
        unwrapped = self._get_par_env(worker=worker, base_env=base_env)
        if unwrapped is not None and hasattr(unwrapped, "base_env"):
            unwrapped = unwrapped.base_env  # FireParallelEnv -> MultiAgentFireEnv (has .H/.W)

        total_cells = 1
        if unwrapped is not None and hasattr(unwrapped, "H"):
            total_cells = unwrapped.H * unwrapped.W
            coverage = float(cells_visited) / max(1, total_cells)

        # Collisions are not currently exposed by the env; set 0 if unavailable
        wall_collisions = float(info.get("wall_collisions", 0)) if isinstance(info, dict) else 0.0
        agent_collisions = float(info.get("agent_collisions", 0)) if isinstance(info, dict) else 0.0

        # Write custom metrics
        episode.custom_metrics["total_reward"] = float(total_reward)
        episode.custom_metrics["episode_length"] = float(length)
        episode.custom_metrics["success"] = 1.0 if success else 0.0
        episode.custom_metrics["fire_found"] = 1.0 if fire_found else 0.0
        episode.custom_metrics["all_agents_reached_fire"] = 1.0 if all_agents_reached_fire else 0.0
        episode.custom_metrics["coverage"] = float(coverage)
        episode.custom_metrics["wall_collisions"] = float(wall_collisions)
        episode.custom_metrics["agent_collisions"] = float(agent_collisions)
        episode.custom_metrics["max_temp"] = float(episode_max_temp)

        # For convenience also log mean reward
        episode.user_data.setdefault("rewards", []).append(total_reward)