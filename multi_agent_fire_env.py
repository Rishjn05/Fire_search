"""Grid-based multi-agent firefighting environment.

The environment exposes a compact Gymnasium API used by the existing training,
evaluation, baseline, and PettingZoo wrapper scripts in this repository.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from fire_dynamics import step_fire
from observations import local_observation, lidar_scan


Position = Tuple[int, int]


class MultiAgentFireEnv(gym.Env):
    """Cooperative multi-agent firefighting environment.

    Grid encoding:
        0 = outside, 1 = wall, 2 = door, 3 = floor,
        4 = active fire, 6 = firefighter (labels only; not stored in building_map).

    Actions: 0-3 move (up/down/right/left), 4 = stay. There is no extinguish
    action. Movement into a cell with active fire is blocked (see
    _propose_move). Fire is frozen once any agent finds it.

    Episode ends (terminated, success) the step every agent has had active
    fire within its own local observation radius (see all_agents_reached_fire).
    Otherwise it is truncated at max_steps.

    Information sharing: each agent's own local 5x5 grid always shows fire
    cells within its own view (genuine local sensing). The team-wide
    "nearest fire" compass vector in the observation extras, however, is
    discovery-gated -- it stays at the "unknown" sentinel for every agent
    until self.fire_found flips True (any single agent getting fire within
    its local radius), at which point the true fire coordinate becomes
    visible to every agent's observation from that step onward. This is
    what implements "agent finds fire -> team learns the coordinate" --
    see _get_obs().
    """

    metadata = {"render_modes": ["rgb_array"]}

    ACTIONS: Dict[int, Position] = {
        0: (-1, 0),  # up
        1: (1, 0),  # down
        2: (0, 1),  # right
        3: (0, -1),  # left
    }
    STAY_ACTION = 4
    NUM_ACTIONS = 5
    # One-time terminal bonus when every agent has reached the fire.
    ALL_REACHED_BONUS = 10.0
    # Memory/exploration additions (see _mark_seen, _memory_features).
    MEM_FEATS = 7
    RECENT_WINDOW = 50          # was 10: loops longer than 10 steps were never penalized
    REVISIT_PENALTY = 0.01      # per earlier visit inside the window, capped at 5x
    NEWLY_SEEN_BONUS = 0.002    # per cell newly inside any agent's 5x5 view

    # Floor used to normalize real CFD temperature (deg C) into the 0-1
    # fire_intensity scale -- see _apply_cfd_frame(). Every T_slices.npy
    # in the uploaded datasets starts at ~29.9-30.0C ambient, so this is
    # a dataset-wide constant rather than something read per-scenario.
    AMBIENT_TEMP_C = 30.0

    # Ignition intensity for the single starting cell, for BOTH the
    # synthetic (no-CFD-data) path and the new CFD-ramp path below -- 1.0
    # matches what the synthetic branch already used, so this only changes
    # behavior for CFD-driven scenarios, which previously skipped a point
    # start entirely (see _ignite_initial_fire).
    IGNITION_POINT_INTENSITY = 1.0

    # ---- Dataset usage (ported from the earlier single-agent project) ----
    # fire_source.npy is stored vertically flipped relative to T_slices.npy
    # (verified: T at the flipped ignition cell is hot at frame 0 for all 15
    # sims in dataset_multi_new). The earlier project flipped it on load and
    # left T_slices untouched; same here.
    FLIP_FIRE_SOURCE = True
    # The earlier project also flipped obstacles.npy. The hot-cell overlap
    # check on dataset_multi_new was mixed (raw 117/165 hot cells on free
    # floor vs flipped 87/165), so this is a switch. Set to True to match the earlier project;
    # a render of sims 005/008/010/015 showed the fire landing in walls and
    # spreading to 69-105 cells with True, so re-check if fire looks wrong.
    FLIP_OBSTACLES = False
    # Time mapping, same as the earlier project: 31 frames, one every 10 s
    # of simulated time, the team "enters" at 60 s (frame 6), and each env
    # step advances 0.4 s. frame = min(round((step*dt + entry)/interval), last).
    SIM_STEP_DT_S = 0.4
    CFD_FRAME_INTERVAL_S = 10.0
    CFD_ENTRY_TIME_S = 60.0
    # A cell counts as "fire" only when its temperature is at least this
    # fraction of the scenario's (min..max) range -- the earlier project's
    # hot_threshold was 50% of temp_max. Anything lower is just heated gas:
    # at the old 0.05 cut, 1400-1572 of 1572 floor cells were "on fire" for
    # nearly the whole episode.
    CFD_FIRE_FRACTION = 0.5
    # Active cells are given at least this intensity so every "fire" cell
    # stays above the 0.05 active threshold the rest of the env uses.
    CFD_MIN_ACTIVE_INTENSITY = 0.1

    def __init__(
        self,
        dataset_dir: Union[str, List[str]] = "dataset_multi_new",
        mode: str = "train",
        num_agents: int = 4,
        max_steps: int = 600,
        building_id: Optional[str] = "7",
        obs_radius: int = 2,
        map_size: int = 32,
        training: bool = True,
        seed: Optional[int] = None,
        lidar_rays: int = 8,
        lidar_range: int = 8,
        append_global_state: bool = True,
        thermal_rays: int = 16,
        random_num_agents: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Create a multi-agent fire environment.

        Args:
            dataset_dir: Kept for compatibility with older scripts. May be
                a single dataset directory (str) or a list of dataset
                directories. When a list is given, every dataset is
                preloaded once at construction time and a fresh building
                is drawn from the pool on every ``reset()`` (see
                ``options={"building_dir": ...}`` to pin one explicitly,
                e.g. for deterministic evaluation loops). This is what
                enables multi-building generalization training instead of
                overfitting to a single layout.
            mode: Experiment split label, e.g. ``train`` or ``test``.
            num_agents: Number of firefighter agents.
            max_steps: Maximum episode length.
            building_id: Optional folder under ``buildings/`` containing
                ``map.npy``. Used as a fallback for any dataset dir that
                doesn't have CFD data on disk. If missing too, a
                procedural building is generated.
            obs_radius: Radius of local observation, 2 gives a 5x5 view.
            map_size: Size for generated square maps.
            training: Kept for compatibility with evaluation scripts.
            seed: Optional random seed.
            lidar_rays: Number of evenly-spaced rays cast outward from each
                agent to sense wall/boundary distance (see observations.lidar_scan).
            lidar_range: Max cells each lidar ray travels before it's
                considered unobstructed (reading clamps to 1.0 beyond this).
            thermal_rays: Number of thermal-camera rays per agent (0 disables).
                Each ray returns the HOTTEST normalized reading along its
                direction, out to (but NOT including) the first wall --
                heat behind a wall is no longer visible, so agents are not
                pulled into pressing against walls --  unlimited range. This
                is the pre-discovery search signal: heat is visible across
                a room/corridor, so agents can move toward it before any
                fire cell enters their 5x5 view. See _thermal_field().
            append_global_state: If True (default), the centralized-critic
                state vector (see get_global_state) is appended to the end
                of EVERY agent's observation row, so state t is always
                aligned with observation t at both sampling and training
                time (no callback injection needed). The actor must use
                only the first ``agent_obs_dim`` values; the critic uses
                all ``agent_obs_dim + state_dim``.
        """
        super().__init__()
        self.mode = mode
        self.num_agents = int(num_agents)
        # (lo, hi): draw the team size uniformly from [lo, hi] every reset().
        self.random_num_agents = random_num_agents
        self.max_steps = int(max_steps)
        self.building_id = building_id
        self.local_grid_radius = int(obs_radius)
        self.local_grid_channels = 4
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)

        self.current_scenario_id: Optional[str] = None
        self.walkable_values = {0, 2, 3, 5}

        dataset_dirs = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        if not dataset_dirs:
            raise ValueError("dataset_dir must be a non-empty string or list of strings")

        # Preload every building in the pool once so per-episode building
        # swaps (see reset()) don't re-hit disk each time. The cached
        # payload per dataset is just a small int32 grid + a scenario-id
        # dict, so keeping several resident per env-runner is cheap.
        self._buildings: Dict[str, dict] = {
            d: self._preload_building(d, map_size) for d in dataset_dirs
        }
        self.dataset_dir_pool: List[str] = list(self._buildings.keys())
        self._select_building(self.dataset_dir_pool[0])

        # extras layout: [y, x, door_y, door_x, fire_y, fire_x,
        #                 total_fire, step_frac, blocked_last_step]
        # `blocked_last_step` is a 0/1 flag set whenever the agent's
        # previous action failed to change its position (wall collision,
        # agent collision, or an invalid action id). The policy network
        # here is a plain feed-forward net with no recurrent memory, and
        # position/local-grid observations are identical before and after
        # a blocked move -- without this flag a deterministic policy has
        # no signal that its last choice failed and will repeat the same
        # action against the same wall indefinitely.
        self.local_grid_channels = 4
        self.lidar_rays = int(lidar_rays)
        self.lidar_range = int(lidar_range)
        self.thermal_rays = int(thermal_rays)
        # +7 memory features: visit counts of the 4 neighbour cells, plus
        # (dy, dx, dist) toward the nearest cell no agent has seen yet.
        self._num_extras = 9 + self.lidar_rays + self.thermal_rays + self.MEM_FEATS
        if self.thermal_rays > 0:
            _ang = np.linspace(0.0, 2.0 * np.pi, self.thermal_rays, endpoint=False)
            # y-down grid, same convention as lidar: theta=0 -> +x, pi/2 -> +y
            self._th_dy = np.sin(_ang).astype(np.float32)
            self._th_dx = np.cos(_ang).astype(np.float32)
        obs_width = self.local_grid_channels * (2 * self.local_grid_radius + 1) ** 2 + self._num_extras
        # Per-agent (actor) observation width, and centralized-critic state
        # width. get_global_state() returns a FIXED 14 values regardless of
        # num_agents (team aggregates, not per-agent slots), so the same
        # trained policy accepts any team size without retraining.
        self.agent_obs_dim = int(obs_width)
        self._unseen_cache = None
        self.append_global_state = bool(append_global_state)
        self.state_dim = 14
        total_width = self.agent_obs_dim + (self.state_dim if self.append_global_state else 0)
        self._total_width = int(total_width)
        self.action_space = spaces.MultiDiscrete([self.NUM_ACTIONS] * self.num_agents)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_agents, total_width),
            dtype=np.float32,
        )

        self.agent_positions: List[Position] = []
        self.trajectory_history: List[List[Position]] = []
        self.shared_visited = np.zeros((self.H, self.W), dtype=np.int32)
        self.shared_seen = np.zeros((self.H, self.W), dtype=np.int32)
        self.fire_intensity = np.zeros((self.H, self.W), dtype=np.float32)
        # Real CFD temperature time series for the scenario ignited this
        # episode (shape (num_frames, H, W), None if the picked scenario
        # has no T_slices.npy -- see _preload_building/_ignite_initial_fire).
        # When set, fire_intensity each step comes from _apply_cfd_frame()
        # instead of the synthetic step_fire() spread model.
        self.current_T_slices: Optional[np.ndarray] = None
        self.current_temp_max = 400.0
        self.current_temp_min = self.AMBIENT_TEMP_C
        self._cfd_cut = self.CFD_FIRE_FRACTION
        self._cfd_frame_idx = 0
        self.fire_found = False
        # Terminal success: true once EVERY agent has, at some point in the
        # episode, had an active fire cell within its own local observation
        # radius. Only checked after fire_found is True.
        self.all_agents_reached_fire = False
        self._agent_reached_fire: List[bool] = [False] * self.num_agents
        self.previous_min_dist: Optional[np.ndarray] = None  # per-agent distances
        self.step_count = 0
        self.last_total_fire = 0.0
        self.episode_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.wall_collisions = 0
        self.agent_collisions = 0
        self.fire_blocked_moves = 0
        self.invalid_actions = 0
        self.last_blocked: List[bool] = [False] * self.num_agents
        # Short-term per-agent visit history used for the anti-oscillation
        # penalty in step() -- separate from `shared_visited`, which is a
        # permanent whole-episode counter used for the novelty bonus.
        self.recent_positions: List[deque] = [deque(maxlen=self.RECENT_WINDOW) for _ in range(self.num_agents)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return observations plus episode info."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}

        n_req = options.get("num_agents")
        if n_req is None and self.random_num_agents is not None:
            lo, hi = self.random_num_agents
            n_req = int(self.rng.integers(int(lo), int(hi) + 1))
        if n_req is not None:
            self.num_agents = int(n_req)

        # Draw a fresh building for this episode. Training leaves this to
        # random.choice over the preloaded pool; evaluation harnesses can
        # pin a specific dataset via options={"building_dir": ...}, mirroring
        # the existing scenario_id pin below.
        building_choice = options.get("building_dir")
        if building_choice is None:
            building_choice = str(self.rng.choice(self.dataset_dir_pool))
        if building_choice not in self._buildings:
            raise ValueError(
                f"building_dir={building_choice!r} not in preloaded pool "
                f"{self.dataset_dir_pool}"
            )
        self._select_building(building_choice)

        self.step_count = 0
        # Buildings can differ in H/W, so these are reallocated (not just
        # cleared) every reset rather than using .fill(0) on the old shape.
        self.shared_visited = np.zeros((self.H, self.W), dtype=np.int32)
        self.fire_intensity = np.zeros((self.H, self.W), dtype=np.float32)
        self.shared_seen = np.zeros((self.H, self.W), dtype=np.int32)
        self._unseen_cache = None
        self.fire_found = False
        self.all_agents_reached_fire = False
        self._agent_reached_fire = [False] * self.num_agents
        self.episode_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.wall_collisions = 0
        self.agent_collisions = 0
        self.fire_blocked_moves = 0
        self.invalid_actions = 0
        self.last_blocked = [False] * self.num_agents
        self.recent_positions = [deque(maxlen=self.RECENT_WINDOW) for _ in range(self.num_agents)]

        self.agent_positions = self._spawn_agents()
        self.trajectory_history = [[pos] for pos in self.agent_positions]
        for y, x in self.agent_positions:
            self.shared_visited[y, x] += 1
        self._mark_seen()

        fire_count = int(options.get("fire_count", 1))
        # scenario_id lets a caller (e.g. an evaluation harness) pin which
        # CFD scenario ignites this episode, instead of always drawing
        # randomly from self.rng. This matters because agent spawn points
        # are fully deterministic (corner-based, not seeded) — so with a
        # small dataset_scenarios pool, random draws can coincidentally
        # repeat the same scenario across "different" episodes, producing
        # literally identical trajectories under a deterministic policy.
        self._ignite_initial_fire(fire_count=fire_count, scenario_id=options.get("scenario_id"))
        self.last_total_fire = float(np.sum(self.fire_intensity))

        # Distance shaping only starts once the fire is found (see step()), so
        # nothing about the true fire position is baked in at reset.
        self.previous_min_dist = None

        return self._get_obs(), self._info()

    def step(self, actions):
        """Advance the simulation by one step.

        Agents move (0-3) or stay (4). Before fire_found the fire evolves
        (CFD frames or synthetic spread); once found it is frozen. The
        episode terminates with success the step all agents have reached
        the fire.

        Args:
            actions: Array-like action for each agent.

        Returns:
            observations, rewards, terminated, truncated, info.
        """
        self.step_count += 1
        actions = np.asarray(actions, dtype=int).reshape(-1)
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        rewards = np.full(self.num_agents, -0.01, dtype=np.float32)

        # BFS movement override: once ANY agent has found the fire, every
        # OTHER agent that hasn't personally come within range of it yet
        # (self._agent_reached_fire[agent_id] still False) has its chosen
        # movement action replaced with the greedy-shortest-path direction
        # toward the fire, from a single shared flood fill (see
        # _fire_distance_map). This is the "other agents now know the
        # coordinate and head straight there" behavior -- scripted rather
        # than RL-learned, since post-discovery routing doesn't need to be
        # learned once the destination is known. An agent that's already
        # within range keeps full RL control (including STAY).
        # Only movement actions (0-3) are ever substituted in -- an agent
        # not yet in range that chose STAY still gets rerouted, since
        # standing still doesn't help it get there.
        if self.fire_found and not all(self._agent_reached_fire):
            fire_dist = self._fire_distance_map()
            if np.isfinite(fire_dist).any():
                for agent_id in range(self.num_agents):
                    if self._agent_reached_fire[agent_id]:
                        continue
                    y, x = self.agent_positions[agent_id]
                    best_action = None
                    best_dist = fire_dist[y, x] if self._in_bounds((y, x)) else np.inf
                    for act_id, (dy, dx) in self.ACTIONS.items():
                        ny, nx = y + dy, x + dx
                        if self._in_bounds((ny, nx)) and fire_dist[ny, nx] < best_dist:
                            best_dist = fire_dist[ny, nx]
                            best_action = act_id
                    if best_action is not None:
                        actions[agent_id] = best_action
                    # else: no neighbor improves on the current cell's
                    # distance (already as close as the grid allows, or
                    # unreachable) -- leave the RL-chosen action as-is.

        proposed = list(self.agent_positions)
        occupied = set(self.agent_positions)
        reserved: set[Position] = set()
        blocked = [False] * self.num_agents
        for agent_id, action in enumerate(actions):
            if action in self.ACTIONS:
                proposed[agent_id], penalty = self._propose_move(
                    self.agent_positions[agent_id],
                    self.ACTIONS[int(action)],
                    occupied,
                    reserved,
                )
                reserved.add(proposed[agent_id])
                rewards[agent_id] += penalty
                # penalty < 0 here means the move was rejected (wall or
                # agent collision) and the agent's position didn't change.
                blocked[agent_id] = penalty < 0.0
            elif action == self.STAY_ACTION:
                rewards[agent_id] -= 0.005  # stay
                blocked[agent_id] = False  # intentional stay, not a failure
            else:
                self.invalid_actions += 1
                rewards[agent_id] -= 0.10
                blocked[agent_id] = True

        self.last_blocked = blocked
        self.agent_positions = proposed
        for agent_id, pos in enumerate(self.agent_positions):
            self.trajectory_history[agent_id].append(pos)
            # Novelty bonus: reward the team the first time ANY agent
            # reaches a cell (shared_visited is shared, not per-agent),
            # so agents are pushed to spread into unexplored rooms rather
            # than re-tread ground a teammate already covered. Anti-
            # oscillation penalty: if this cell is already-visited AND was
            # visited by *this* agent within its last 10 steps, penalize --
            # this targets short back-and-forth loops (e.g. an agent stuck
            # ping-ponging in a dead-end room) that the novelty bonus alone
            # doesn't discourage since revisits elsewhere aren't penalized.
            if self.shared_visited[pos] == 0:
                rewards[agent_id] += 0.03
            else:
                # Scaled by how often this agent was here in the last
                # RECENT_WINDOW steps, so long loops get progressively costlier.
                n_recent = sum(1 for q in self.recent_positions[agent_id] if q == pos)
                if n_recent:
                    rewards[agent_id] -= self.REVISIT_PENALTY * min(n_recent, 5)
            self.recent_positions[agent_id].append(pos)
            self.shared_visited[pos] += 1
        new_seen = self._mark_seen()
        rewards += np.float32(self.NEWLY_SEEN_BONUS) * new_seen.astype(np.float32)

        # Fire evolves each step, gated on self.fire_found:
        # 1) BEFORE found: fire grows on its own -- from the real CFD
        #    T_slices sequence when the scenario has one, otherwise from the
        #    synthetic spread/diffusion model.
        # 2) AFTER found: fire is frozen where it was, so the team converges
        #    on a static target. No decay phase: the episode ends the step
        #    the last agent reaches the fire.
        if not self.fire_found:
            if self.current_T_slices is not None:
                self._cfd_frame_idx = self._cfd_frame_for_step(self.step_count)
                self._apply_cfd_frame()
            else:
                self.fire_intensity = step_fire(
                    self.fire_intensity,
                    self._fire_blocking_map(),
                    spread_rate=0.025,
                    decay_rate=0.0,
                    diffusion_rate=0.04,
                ).astype(np.float32)
                self.fire_intensity[self.building_map == 1] = 0.0
                self.fire_intensity[self.building_map == 0] = 0.0

        current_fire = float(np.sum(self.fire_intensity))

        # Once fire is found, charge a small per-step cost proportional to the
        # fraction of the grid on fire (bounded at -0.1/step) so idle waiting
        # is not free and the team is pushed to converge quickly.
        if self.fire_found:
            active_fire_cells = int(np.count_nonzero(self.fire_intensity > 0.05))
            total_cells = max(1, self.H * self.W)
            fire_fraction = active_fire_cells / total_cells
            rewards -= np.float32(0.1) * fire_fraction

        # Dense shaping: reward EACH agent individually for closing distance
        # to its own nearest active fire cell, not a single team-wide
        # scalar. Previously this used one shared `min(...)` across all
        # agents, so the instant ANY one agent reached the fire, the "team
        # minimum" bottomed out near 0 and every other agent's reward delta
        # collapsed to ~0 regardless of which direction it moved -- there
        # was no incentive left for a second/third agent to travel toward
        # fire once a teammate got there first, so they'd drift off to the
        # cheap novelty-exploration bonus instead of converging to help
        # reach it. Tracking previous_min_dist per-agent fixes that: an
        # agent far from the fire keeps collecting its own distance-closing
        # reward all the way in, independent of a teammate already there.
        fire_cells = [tuple(p) for p in np.argwhere(self.fire_intensity > 0.05)]
        if self.fire_found and fire_cells and self.previous_min_dist is not None:
            new_min_dist = np.zeros(self.num_agents, dtype=np.float32)
            has_los = np.zeros(self.num_agents, dtype=bool)
            for i, pos in enumerate(self.agent_positions):
                nearest_fire = min(
                    fire_cells,
                    key=lambda f: abs(pos[0] - f[0]) + abs(pos[1] - f[1]),
                )
                new_min_dist[i] = abs(pos[0] - nearest_fire[0]) + abs(pos[1] - nearest_fire[1])
                has_los[i] = self._has_line_of_sight(pos, nearest_fire)
            delta = self.previous_min_dist - new_min_dist
            # Manhattan distance is a straight-line shortcut that ignores
            # walls entirely -- it's only a trustworthy shaping signal when
            # nothing actually blocks that line (see _has_line_of_sight).
            # When a wall sits between the agent and its nearest fire cell,
            # delta reflects progress toward an unwalkable shortcut, not a
            # walkable path: it can reward standing still (distance frozen
            # at the wall) and penalize the very detour needed to actually
            # get there. Zeroing shaping in that case removes the false
            # signal without claiming to know which way to detour instead
            # -- exploration (the novelty bonus) still has to find that.
            rewards += np.where(has_los, np.float32(0.02) * delta, 0.0)
            self.previous_min_dist = new_min_dist

        # Detection: success the moment any agent has an active fire cell
        # within its local observation radius.
        newly_found = False
        if not self.fire_found:
            for (y, x) in self.agent_positions:
                for (fy, fx) in fire_cells:
                    if max(abs(y - fy), abs(x - fx)) <= self.local_grid_radius:
                        self.fire_found = True
                        newly_found = True
                        break
                if self.fire_found:
                    break
        if newly_found:
            rewards += 5.0
            # Baseline for the distance-closing shaping, taken now so the
            # first shaped step doesn't see a stale/huge delta.
            self.previous_min_dist = self._current_min_dist(fire_cells)

        # Every agent (not just one) has had an active fire cell within its
        # own local observation radius at some point since fire_found. The
        # per-agent flag is OR-ed in each step. This is the terminal success.
        if self.fire_found and not self.all_agents_reached_fire and fire_cells:
            for agent_id, (y, x) in enumerate(self.agent_positions):
                if self._agent_reached_fire[agent_id]:
                    continue
                for (fy, fx) in fire_cells:
                    if max(abs(y - fy), abs(x - fx)) <= self.local_grid_radius:
                        self._agent_reached_fire[agent_id] = True
                        break
            if all(self._agent_reached_fire):
                self.all_agents_reached_fire = True
                rewards += self.ALL_REACHED_BONUS

        # Success = all agents reached the fire. Terminal.
        terminated = self.all_agents_reached_fire
        truncated = self.step_count >= self.max_steps

        self.last_total_fire = current_fire
        self.episode_reward += rewards
        return self._get_obs(), rewards, bool(terminated), bool(truncated), self._info()

    def render(self) -> np.ndarray:
        """Render the current state as an RGB array."""
        colors = {
            0: np.array([224, 237, 245], dtype=np.uint8),  # outside
            1: np.array([28, 30, 32], dtype=np.uint8),  # wall
            2: np.array([130, 78, 28], dtype=np.uint8),  # door
            3: np.array([199, 204, 199], dtype=np.uint8),  # floor
        }
        image = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for value, color in colors.items():
            image[self.building_map == value] = color
        fire = np.clip(self.fire_intensity, 0.0, 1.0)
        # Align the visible threshold with the threshold used elsewhere for
        # "active fire" (detection/reward logic uses > 0.05). Cells below
        # that aren't gameplay-relevant, so no need to render faint embers.
        fire_cells = fire > 0.05
        if np.any(fire_cells):
            heat = fire[fire_cells, None]
            # Floor the blend weight so any active fire cell stays clearly
            # visible even as its true intensity decays toward the
            # detection threshold, instead of washing out into the
            # background color.
            heat = np.clip(heat, 0.45, 1.0)
            fire_color = np.array([255, 70, 20], dtype=np.float32)
            image[fire_cells] = (
                image[fire_cells].astype(np.float32) * (1.0 - heat)
                + fire_color * heat
            ).astype(np.uint8)

        for path in self.trajectory_history:
            for y, x in path[-80:]:
                image[y, x] = np.array([44, 125, 210], dtype=np.uint8)

        for y, x in self.agent_positions:
            image[y, x] = np.array([20, 85, 170], dtype=np.uint8)
            for dy, dx in self.ACTIONS.values():
                ny, nx = y + dy, x + dx
                if self._in_bounds((ny, nx)) and self.building_map[ny, nx] != 1:
                    image[ny, nx] = np.array([70, 150, 230], dtype=np.uint8)

        scale = max(4, min(10, 500 // max(self.H, self.W)))
        frame = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        return frame

    def close(self) -> None:
        """Close rendering resources."""
        plt.close("all")

    def get_global_state(self) -> np.ndarray:
        """Return a normalized centralized state vector for MAPPO critics."""
        hw = max(1, self.H * self.W)
        active = self.fire_intensity > 0.05
        n_active = int(np.count_nonzero(active))
        if n_active:
            ys, xs = np.nonzero(active)
            cy = float(ys.mean()) / max(1, self.H - 1)
            cx = float(xs.mean()) / max(1, self.W - 1)
        else:
            cy = cx = 0.0
        # Ground truth is fine here: the critic is centralized and only
        # used in training, so it may see the fire location even before
        # any agent has found it.
        state = [
            float(self.step_count) / max(1, self.max_steps),
            n_active / hw,
            float(np.sum(self.fire_intensity)) / hw,
            float(np.max(self.fire_intensity)) if self.fire_intensity.size else 0.0,
            1.0 if self.fire_found else 0.0,
            1.0 if self.all_agents_reached_fire else 0.0,
            cy,
            cx,
            min(1.0, int(np.count_nonzero(self.shared_visited)) / max(1, len(self.floor_cells))),  # explored fraction
        ]
        # Team aggregates (permutation-invariant, independent of team size):
        # fraction of agents that have reached the fire, mean agent position,
        # mean distance to the fire centroid (Chebyshev, normalized), and
        # team size / 8. Per-agent slots would make the observation width
        # depend on num_agents and tie a trained policy to one team size.
        n = max(1, len(self.agent_positions))
        pos = np.asarray(self.agent_positions, dtype=np.float32).reshape(-1, 2)
        mean_y = float(pos[:, 0].mean()) / max(1, self.H - 1) if len(pos) else 0.0
        mean_x = float(pos[:, 1].mean()) / max(1, self.W - 1) if len(pos) else 0.0
        if n_active and len(pos):
            fy = cy * max(1, self.H - 1)
            fx = cx * max(1, self.W - 1)
            d = np.maximum(np.abs(pos[:, 0] - fy), np.abs(pos[:, 1] - fx))
            mean_dist = float(d.mean()) / max(1, max(self.H, self.W))
        else:
            mean_dist = 0.0
        state.extend([
            sum(1 for r in self._agent_reached_fire if r) / n,
            mean_y,
            mean_x,
            mean_dist,
            n / 8.0,
        ])
        return np.clip(np.asarray(state, dtype=np.float32), 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.num_agents, self._total_width), dtype=np.float32)
        unseen_dist = self._unseen_distance_map()
        # Computed once per call so every agent's row carries the identical
        # state, taken at the same moment as the observation itself.
        global_state = self.get_global_state() if self.append_global_state else None
        thermal = self._thermal_scan() if self.thermal_rays > 0 else None
        for agent_id, (y, x) in enumerate(self.agent_positions):
            local = local_observation(self, agent_id, self.local_grid_radius)
            nearest_door = self._nearest_normalized((y, x), self.doors)
            # Discovery-gated fire compass: before ANY agent has found fire
            # (self.fire_found), every agent's own local 5x5 grid is still
            # the only way it can sense fire, exactly like it can't sense
            # anything else outside local_grid_radius. This field used to
            # be computed from the ground-truth fire_intensity map
            # unconditionally, which gave every agent a compass pointing
            # at the fire from step 0 regardless of whether anyone had
            # actually found it -- that skipped the search phase entirely
            # and made "cooperative discovery" a no-op, since nothing
            # meaningful was gated on fire_found except the reward bonus.
            # Passing an empty target list once fire_found flips True
            # reuses _nearest_normalized's existing "no info" convention
            # ((0.0, 0.0), same as when there are no doors/no fire) rather
            # than inventing a new sentinel value.
            fire_cells_for_obs = (
                [tuple(p) for p in np.argwhere(self.fire_intensity > 0.05)]
                if self.fire_found
                else []
            )
            nearest_fire = self._nearest_normalized((y, x), fire_cells_for_obs)
            extras = np.asarray(
                [
                    y / max(1, self.H - 1),
                    x / max(1, self.W - 1),
                    nearest_door[0],
                    nearest_door[1],
                    nearest_fire[0],
                    nearest_fire[1],
                    float(np.sum(self.fire_intensity)) / max(1, self.H * self.W),
                    float(self.step_count) / max(1, self.max_steps),
                    1.0 if self.last_blocked[agent_id] else 0.0,
                ],
                dtype=np.float32,
            )
            lidar = lidar_scan(
                self, agent_id, num_rays=self.lidar_rays, max_range=self.lidar_range
            )
            parts = [local, extras, lidar, self._memory_features((y, x), unseen_dist)]
            if thermal is not None:
                parts.append(thermal[agent_id])
            if self.append_global_state:
                parts.append(global_state)
            obs[agent_id] = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _current_min_dist(self, fire_cells) -> Optional[np.ndarray]:
        if not fire_cells:
            return None
        return np.asarray(
            [min(abs(p[0] - fy) + abs(p[1] - fx) for fy, fx in fire_cells) for p in self.agent_positions],
            dtype=np.float32,
        )

    def _mark_seen(self) -> np.ndarray:
        """Mark every cell inside each agent's local view as seen (team-shared).

        Returns the number of newly-seen cells credited to each agent.
        """
        r = self.local_grid_radius
        gained = np.zeros(self.num_agents, dtype=np.int32)
        for i, (y, x) in enumerate(self.agent_positions):
            win = self.shared_seen[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1]
            gained[i] = int(np.count_nonzero(win == 0))
            win[...] = 1
        return gained

    def _unseen_distance_map(self) -> np.ndarray:
        """BFS distance (walkable cells only) to the nearest never-seen cell.

        Cached on the number of seen cells, so it is only recomputed when the
        team actually uncovers something new.
        """
        key = int(np.count_nonzero(self.shared_seen))
        if self._unseen_cache is not None and self._unseen_cache[0] == key:
            return self._unseen_cache[1]
        dist = np.full((self.H, self.W), np.inf, dtype=np.float32)
        free = self.building_map != 1
        src = free & (self.shared_seen == 0)
        q: deque = deque()
        for y, x in np.argwhere(src):
            dist[y, x] = 0.0
            q.append((int(y), int(x)))
        while q:
            y, x = q.popleft()
            d = dist[y, x] + 1.0
            for dy, dx in self.ACTIONS.values():
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W and free[ny, nx] and dist[ny, nx] > d:
                    dist[ny, nx] = d
                    q.append((ny, nx))
        self._unseen_cache = (key, dist)
        return dist

    def _memory_features(self, pos: Position, unseen_dist: np.ndarray) -> np.ndarray:
        y, x = pos
        out = np.zeros(self.MEM_FEATS, dtype=np.float32)
        best, best_d = None, unseen_dist[y, x]
        for k, (act, (dy, dx)) in enumerate(self.ACTIONS.items()):
            ny, nx = y + dy, x + dx
            if not self._in_bounds((ny, nx)) or self.building_map[ny, nx] == 1:
                out[k] = 1.0
                continue
            out[k] = min(int(self.shared_visited[ny, nx]), 5) / 5.0
            if unseen_dist[ny, nx] < best_d:
                best_d, best = unseen_dist[ny, nx], (dy, dx)
        if best is not None:
            out[4], out[5] = (best[0] + 1) / 2.0, (best[1] + 1) / 2.0
        else:
            out[4] = out[5] = 0.5
        d_here = unseen_dist[y, x]
        out[6] = 1.0 if not np.isfinite(d_here) else min(float(d_here), 60.0) / 60.0
        return out

    def _thermal_field(self) -> np.ndarray:
        """(H, W) float32 in [0, 1]: what a thermal camera would read.

        BEFORE the fire is found: the live CFD temperature frame (the same
        frame that drives fire_intensity), normalized by this scenario's
        (temp_min..temp_max) range and passed through sqrt. The sqrt matters:
        most of the map sits only a few degrees above ambient while the hot
        core is hundreds of degrees, so a linear scale leaves the far-field
        gradient near zero. It keeps the heated-gas halo around the fire
        visible, which is what gives a search signal. Scenarios without CFD
        data fall back to fire_intensity.

        AFTER the fire is found: the live fire_intensity map (the CFD frame
        is frozen at that point).
        """
        if self.fire_found or self.current_T_slices is None:
            return np.clip(self.fire_intensity, 0.0, 1.0).astype(np.float32)
        frame = self.current_T_slices[self._cfd_frame_idx]
        denom = max(1e-6, self.current_temp_max - self.current_temp_min)
        field = np.sqrt(np.clip((frame - self.current_temp_min) / denom, 0.0, 1.0))
        field = field.astype(np.float32)
        return field

    def _thermal_scan(self) -> np.ndarray:
        """(num_agents, thermal_rays): max thermal reading along each ray.

        Rays start at the agent's cell (its own reading is the floor), march
        outward with no range limit and stop BEFORE the first wall cell. Out-of-bounds stops a ray without contributing.
        """
        field = self._thermal_field()
        R = self.thermal_rays
        L = max(self.H, self.W)
        steps = np.arange(1, L + 1, dtype=np.float32)[None, :]
        idx = np.arange(L)[None, :]
        out = np.zeros((self.num_agents, R), dtype=np.float32)
        for a, (y0, x0) in enumerate(self.agent_positions):
            yy = np.rint(y0 + self._th_dy[:, None] * steps).astype(np.int32)
            xx = np.rint(x0 + self._th_dx[:, None] * steps).astype(np.int32)
            inb = (yy >= 0) & (yy < self.H) & (xx >= 0) & (xx < self.W)
            yc = np.clip(yy, 0, self.H - 1)
            xc = np.clip(xx, 0, self.W - 1)
            wall = (self.building_map[yc, xc] == 1) | ~inb
            has = wall.any(axis=1)
            first = np.where(has, wall.argmax(axis=1), L)
            include = idx < first[:, None]
            vals = np.where(include, field[yc, xc], 0.0)
            out[a] = np.maximum(vals.max(axis=1), field[y0, x0])
        return out

    def _info(self) -> dict:
        total_fire = float(np.sum(self.fire_intensity))
        return {
            "step": self.step_count,
            "dataset_dir": self.dataset_dir,
            "scenario_id": self.current_scenario_id,
            "positions": list(self.agent_positions),
            "fire_remaining": total_fire,
            "fire_found": self.fire_found,
            "all_agents_reached_fire": self.all_agents_reached_fire,
            "success": self.all_agents_reached_fire,
            "cells_visited": int(np.count_nonzero(self.shared_visited)),
            "burned_cells": int(np.count_nonzero(self.fire_intensity > 0.05)),
            "episode_max_temp": float(np.max(self.fire_intensity)) if self.fire_intensity.size else 0.0,
            "wall_collisions": int(self.wall_collisions),
            "agent_collisions": int(self.agent_collisions),
            "fire_blocked_moves": int(self.fire_blocked_moves),
            "invalid_actions": int(self.invalid_actions),
            "episode_reward": self.episode_reward.copy(),
        }

    def _preload_building(self, dataset_dir: str, map_size: int) -> dict:
        """Load (or generate) a single building's map and CFD scenarios.

        Pure function of ``dataset_dir`` -- does not mutate ``self`` -- so it
        can be called once per dataset at construction time and cached in
        ``self._buildings``. ``_select_building`` later applies one of these
        cached payloads onto ``self`` at the start of each episode.

        Returns:
            dict with keys ``building_map`` (HxW int array) and
            ``dataset_scenarios`` (scenario_id -> ignition cell + CFD temp
            range, possibly empty if no CFD dataset was found on disk).
        """
        dataset_scenarios: Dict[str, dict] = {}
        dataset_obstacles: Optional[np.ndarray] = None

        dataset_path = Path(dataset_dir)
        metadata_path = dataset_path / "metadata.json"
        obstacles_path = dataset_path / "obstacles.npy"
        if metadata_path.exists() and obstacles_path.exists():
            with metadata_path.open("r") as f:
                metadata = json.load(f)

            dataset_obstacles = np.asarray(np.load(obstacles_path), dtype=np.int32)
            if self.FLIP_OBSTACLES:
                dataset_obstacles = np.flip(dataset_obstacles, axis=0).copy()

            if self.mode == "train":
                scenario_ids = metadata.get("train_split", [])
            elif self.mode == "test":
                scenario_ids = metadata.get("test_split", [])
            else:
                scenario_ids = [s["id"] for s in metadata.get("simulations", [])]

            sim_meta_by_id = {s["id"]: s for s in metadata.get("simulations", [])}

            for sim_id in scenario_ids:
                fire_source_path = dataset_path / sim_id / "fire_source.npy"
                if not fire_source_path.exists():
                    continue
                fire_source = np.load(fire_source_path)
                if self.FLIP_FIRE_SOURCE:
                    fire_source = np.flip(fire_source, axis=0)
                fire_idx = np.argwhere(fire_source == 1)
                if len(fire_idx) == 0:
                    continue
                fire_pos = (int(fire_idx[0][0]), int(fire_idx[0][1]))
                temp = sim_meta_by_id.get(sim_id, {}).get("temperature", {})
                # T_slices.npy shares this sim's (H, W) grid with
                # obstacles.npy/fire_source.npy (confirmed across all five
                # uploaded datasets), one frame per recorded CFD timestep.
                # Present for every sim in every dataset, unlike
                # V_slices.npy (visibility/smoke), which only exists for
                # 9 of 15 sims in dataset_multi_new and nowhere else --
                # that's why only temperature is wired into fire dynamics
                # here, not smoke.
                t_slices_path = dataset_path / sim_id / "T_slices.npy"
                t_slices = (
                    np.load(t_slices_path).astype(np.float32)
                    if t_slices_path.exists()
                    else None
                )
                dataset_scenarios[sim_id] = {
                    "fire_pos": fire_pos,
                    "temp_min": float(temp.get("min", self.AMBIENT_TEMP_C)),
                    "temp_max": float(temp.get("max", 400.0)),
                    "T_slices": t_slices,
                }

        if dataset_obstacles is not None:
            building_map = self._normalize_loaded_map(dataset_obstacles)
        else:
            building_map = None
            if self.building_id is not None:
                map_path = Path("buildings") / str(self.building_id) / "map.npy"
                if map_path.exists():
                    raw = np.asarray(np.load(map_path), dtype=np.int32)
                    building_map = self._normalize_loaded_map(raw)
            if building_map is None:
                building_map = self._generate_building(int(map_size))

        return {
            "building_map": building_map,
            "dataset_scenarios": dataset_scenarios,
        }

    def _select_building(self, dataset_dir: str) -> None:
        """Make a preloaded building (see ``_preload_building``) active.

        Sets ``self.dataset_dir``/``building_map``/``H``/``W``/``doors``/
        ``floor_cells``/``dataset_scenarios``. Callers are responsible for
        re-allocating any HxW-shaped arrays (``shared_visited``,
        ``fire_intensity``) afterwards, since different buildings in the
        pool can have different dimensions.
        """
        cached = self._buildings[dataset_dir]
        self.dataset_dir = dataset_dir
        self.building_map = cached["building_map"]
        self.dataset_scenarios = cached["dataset_scenarios"]
        self._dataset_available = bool(self.dataset_scenarios)
        self.H, self.W = self.building_map.shape
        self.doors = self._find_doors()
        self.floor_cells = self._find_cells([3])
        if not self.floor_cells:
            self.floor_cells = self._find_cells([2, 3])

    def _normalize_loaded_map(self, raw: np.ndarray) -> np.ndarray:
        building = np.asarray(raw, dtype=np.int32).copy()
        if building.ndim != 2:
            raise ValueError("Building map must be a 2D array")

        unique = set(np.unique(building).tolist())
        if unique.issubset({0, 1}):
            # Common dataset format: 1 wall, 0 free. Treat free cells as
            # building floor.
            normalized = np.where(building == 1, 1, 3).astype(np.int32)
        else:
            normalized = building
            normalized[~np.isin(normalized, [0, 1, 2, 3, 4, 5, 6])] = 3
            normalized[normalized == 4] = 3
            normalized[normalized == 6] = 3

        # The outer border is walled off rather than left open, regardless
        # of which branch above ran: leaving it as code 0 ("outside") only
        # stops FIRE from spreading there (_fire_blocking_map treats 0 as
        # blocking) but _propose_move -- the actual agent movement check --
        # only blocks on code 1, so agents could freely walk the entire
        # perimeter ring outside the real rooms. Previously this was only
        # applied in the {0,1}-subset branch above, which silently skipped
        # any dataset whose raw obstacles.npy uses more than two codes (e.g.
        # a door code already baked in) -- confirmed visually on
        # dataset_d10, whose raw format takes the else branch: an agent's
        # whole trajectory hugging the border instead of entering the
        # building interior, for the entire episode, while fire spread
        # unchecked through the interior it never reached.
        normalized[0, :] = 1
        normalized[-1, :] = 1
        normalized[:, 0] = 1
        normalized[:, -1] = 1

        self._ensure_door(normalized)
        return normalized

    def _generate_building(self, map_size: int) -> np.ndarray:
        size = int(np.clip(map_size, 10, 40))
        building = np.zeros((size, size), dtype=np.int32)
        top, bottom = 2, size - 3
        left, right = 2, size - 3
        building[top : bottom + 1, left : right + 1] = 3
        building[top, left : right + 1] = 1
        building[bottom, left : right + 1] = 1
        building[top : bottom + 1, left] = 1
        building[top : bottom + 1, right] = 1

        mid_y = (top + bottom) // 2
        mid_x = (left + right) // 2
        building[mid_y, left + 1 : right] = 1
        building[top + 1 : bottom, mid_x] = 1
        building[mid_y, mid_x] = 3
        building[mid_y, left + 3] = 3
        building[mid_y, right - 3] = 3
        building[top + 3, mid_x] = 3
        building[bottom - 3, mid_x] = 3

        building[bottom, mid_x] = 2
        if size >= 20:
            building[top, mid_x] = 2
        return building

    def _boundary_connected_free(self, free: np.ndarray) -> np.ndarray:
        outside = np.zeros_like(free, dtype=bool)
        stack: List[Position] = []
        h, w = free.shape
        for y in range(h):
            for x in (0, w - 1):
                if free[y, x]:
                    stack.append((y, x))
        for x in range(w):
            for y in (0, h - 1):
                if free[y, x]:
                    stack.append((y, x))

        while stack:
            y, x = stack.pop()
            if outside[y, x]:
                continue
            outside[y, x] = True
            for dy, dx in self.ACTIONS.values():
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not outside[ny, nx]:
                    stack.append((ny, nx))
        return outside

    def _ensure_door(self, building: np.ndarray) -> None:
        if np.any(building == 2):
            return
        floors = np.argwhere(building == 3)
        if floors.size == 0:
            return

        # Prefer an existing floor cell adjacent to outside so agents can enter
        # from the reserved exterior border even when imported maps lack doors.
        for y, x in floors:
            for dy, dx in self.ACTIONS.values():
                ny, nx = int(y + dy), int(x + dx)
                if 0 <= ny < building.shape[0] and 0 <= nx < building.shape[1]:
                    if building[ny, nx] == 0:
                        building[int(y), int(x)] = 2
                        return

        walls = np.argwhere(building == 1)
        if walls.size == 0:
            y, x = floors[0]
            building[int(y), int(x)] = 2
            return

        center = np.mean(floors, axis=0)
        distances = np.sum((walls - center) ** 2, axis=1)
        y, x = walls[int(np.argmax(distances))]
        building[int(y), int(x)] = 2

    def _find_doors(self) -> List[Position]:
        doors = [tuple(map(int, p)) for p in np.argwhere(self.building_map == 2)]
        if doors:
            return doors
        floors = self._find_cells([3])
        return floors[:1]

    def _find_cells(self, values: List[int]) -> List[Position]:
        mask = np.isin(self.building_map, values)
        return [tuple(map(int, p)) for p in np.argwhere(mask)]

    def _largest_connected_component(self, cells: List[Position]) -> List[Position]:
        """Return the largest 4-connected group within ``cells``.

        Used to keep spawn candidates confined to floor space that is
        actually reachable, rather than isolated 1-2 cell pockets that are
        geometrically close to a corner but walled off from the rest of the
        building (which renders as an agent sitting "inside" a wall mass).
        """
        cell_set = set(cells)
        visited: set = set()
        best_component: List[Position] = []
        for start in cells:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component: List[Position] = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in self.ACTIONS.values():
                    nxt = (y + dy, x + dx)
                    if nxt in cell_set and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            if len(component) > len(best_component):
                best_component = component
        return best_component

    def _spawn_agents(self) -> List[Position]:
        """Spawn agents near separate corners of the building's floor area.

        Each agent is assigned to the nearest available floor cell to one of
        the four bounding-box corners of the walkable floor space. If there
        are more agents than corners, agents cycle through corners so they
        still start spread apart rather than stacking on one corner.
        """
        pool = self.floor_cells or self._find_cells([2, 3])
        if not pool:
            pool = self._find_cells(list(self.walkable_values))
        if not pool:
            center = (self.H // 2, self.W // 2)
            return [center] * self.num_agents

        # Restrict to the largest connected walkable region so agents never
        # spawn in a small disconnected pocket that is geometrically near a
        # corner but has no walkable path out (looks like spawning "inside"
        # a wall).
        pool = self._largest_connected_component(pool) or pool

        ys = [p[0] for p in pool]
        xs = [p[1] for p in pool]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)

        # Ordered so consecutive picks are diagonal opposites, not
        # adjacent along one edge. This matters most for num_agents=2:
        # the old top-left/top-right/bottom-left/bottom-right order gave
        # 2-agent runs both agents on the same top edge (corners[0],
        # corners[1]), leaving the whole bottom half of the building
        # equidistant from both spawns. Diagonal ordering means 2 agents
        # get top-left + bottom-right, maximizing initial coverage.
        corners = [
            (min_y, min_x),  # top-left
            (max_y, max_x),  # bottom-right
            (min_y, max_x),  # top-right
            (max_y, min_x),  # bottom-left
        ]

        def nearest_unclaimed(target: Position, taken: set[Position]) -> Position:
            best = None
            best_dist = None
            for cell in pool:
                if cell in taken:
                    continue
                dist = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
                if best_dist is None or dist < best_dist:
                    best = cell
                    best_dist = dist
            if best is None:
                # Every floor cell is claimed (more agents than free cells);
                # allow reuse rather than crashing.
                best = min(
                    pool,
                    key=lambda c: abs(c[0] - target[0]) + abs(c[1] - target[1]),
                )
            return best

        positions: List[Position] = []
        taken: set[Position] = set()
        for i in range(self.num_agents):
            corner = corners[i % len(corners)]
            pos = nearest_unclaimed(corner, taken)
            positions.append(pos)
            taken.add(pos)

        return positions

    @property
    def available_scenario_ids(self) -> List[str]:
        """Sorted CFD scenario ids available for this env's mode/split.

        Exposed so evaluation harnesses can cycle deterministically through
        distinct scenarios (via reset(options={"scenario_id": ...})) rather
        than relying on random draws that can coincidentally repeat.
        """
        return sorted(self.dataset_scenarios.keys())

    def _fire_distance_map(self) -> np.ndarray:
        """Multi-source BFS distance (grid steps) from every passable cell
        to the nearest active fire cell. np.inf where no
        active fire exists or a cell can't reach any (e.g. sealed off by
        walls). Used once per step, shared across all not-yet-arrived
        agents, by the BFS movement override in step() -- computing one
        flood fill from the fire outward is far cheaper than a separate
        path search per agent, and both give the same next-step choice.

        Fire cells themselves are never given a finite distance (agents
        don't step onto active fire -- see _propose_move), so the flood
        fill is seeded at each fire cell's passable neighbors (distance 1)
        rather than at the fire cells themselves.
        """
        dist = np.full((self.H, self.W), np.inf, dtype=np.float32)
        fire_mask = self.fire_intensity > 0.05
        if not np.any(fire_mask):
            return dist
        passable = (self.building_map != 1) & (~fire_mask)
        q: deque = deque()
        for fy, fx in np.argwhere(fire_mask):
            for dy, dx in self.ACTIONS.values():
                ny, nx = int(fy) + dy, int(fx) + dx
                if self._in_bounds((ny, nx)) and passable[ny, nx] and dist[ny, nx] > 1:
                    dist[ny, nx] = 1
                    q.append((ny, nx))
        while q:
            y, x = q.popleft()
            for dy, dx in self.ACTIONS.values():
                ny, nx = y + dy, x + dx
                if self._in_bounds((ny, nx)) and passable[ny, nx] and dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))
        return dist

    def _cfd_frame_for_step(self, step: int) -> int:
        """Frame index for an env step: round((step*dt + entry)/interval),
        clamped to the last frame (earlier project's mapping)."""
        last = self.current_T_slices.shape[0] - 1
        t = step * self.SIM_STEP_DT_S + self.CFD_ENTRY_TIME_S
        return int(min(last, max(0, round(t / self.CFD_FRAME_INTERVAL_S))))

    def _apply_cfd_frame(self) -> None:
        """Set fire_intensity from the current real CFD temperature frame.

        A cell is fire only if its temperature is at least CFD_FIRE_FRACTION
        of the scenario's (temp_min..temp_max) range (self._cfd_cut is that
        fraction, lowered per scenario if the sim never gets that hot on
        walkable cells, so every scenario has some fire). Active cells get
        intensity in [CFD_MIN_ACTIVE_INTENSITY, 1], so everything the rest
        of the env treats as active (> 0.05) is exactly the hot core, not
        the heated gas around it. Walls/outside cells are zeroed.
        """
        frame = self.current_T_slices[self._cfd_frame_idx]
        denom = max(1e-6, self.current_temp_max - self.current_temp_min)
        frac = (frame - self.current_temp_min) / denom
        cut = self._cfd_cut
        scaled = (frac - cut) / max(1e-6, 1.0 - cut)
        lo = self.CFD_MIN_ACTIVE_INTENSITY
        intensity = np.where(frac >= cut, lo + (1.0 - lo) * np.clip(scaled, 0.0, 1.0), 0.0)
        intensity[self.building_map == 1] = 0.0
        intensity[self.building_map == 0] = 0.0
        self.fire_intensity = intensity.astype(np.float32)

    def _ignite_initial_fire(self, fire_count: int, scenario_id: Optional[str] = None) -> None:
        self.current_T_slices = None
        self._cfd_frame_idx = 0
        if self.dataset_scenarios:
            if scenario_id is not None and scenario_id in self.dataset_scenarios:
                self.current_scenario_id = scenario_id
            else:
                self.current_scenario_id = str(self.rng.choice(list(self.dataset_scenarios.keys())))
            scenario = self.dataset_scenarios[self.current_scenario_id]
            self.current_temp_max = scenario["temp_max"]
            self.current_temp_min = scenario["temp_min"]
            t_slices = scenario.get("T_slices")
            y, x = scenario["fire_pos"]
            if t_slices is not None:
                self.current_T_slices = t_slices
                # Per-scenario fire cut: CFD_FIRE_FRACTION, but never above
                # 60% of the hottest walkable-cell reading in the whole
                # sim (e.g. sim_011 only reaches 35% of its range on floor,
                # so a flat 0.5 would give an episode with no fire at all
                # and an instant "success").
                denom = max(1e-6, self.current_temp_max - self.current_temp_min)
                walkable = self.building_map != 1
                walkable &= self.building_map != 0
                peak = float((t_slices[:, walkable].max() - self.current_temp_min) / denom)
                self._cfd_cut = float(min(self.CFD_FIRE_FRACTION, 0.6 * peak))
                self._cfd_frame_idx = self._cfd_frame_for_step(0)
                self._apply_cfd_frame()
                if not np.any(self.fire_intensity > 0.05):
                    # Safety net: seed the (flipped) ignition cell.
                    ignite_at = (y, x) if (self._in_bounds((y, x)) and self.building_map[y, x] != 1) else self._nearest_floor_cell((y, x))
                    if ignite_at is not None:
                        self.fire_intensity[ignite_at] = self.IGNITION_POINT_INTENSITY
                return
            if self._in_bounds((y, x)) and self.building_map[y, x] != 1:
                self.fire_intensity[y, x] = self.IGNITION_POINT_INTENSITY
                return
            fallback = self._nearest_floor_cell((y, x))
            if fallback is not None:
                fy, fx = fallback
                self.fire_intensity[fy, fx] = self.IGNITION_POINT_INTENSITY
                return

        cells = self.floor_cells or self._find_cells([2, 3])
        if not cells:
            cells = [(self.H // 2, self.W // 2)]
        count = min(max(1, fire_count), len(cells))
        indices = self.rng.choice(len(cells), size=count, replace=False)
        for index in np.atleast_1d(indices):
            y, x = cells[int(index)]
            self.fire_intensity[y, x] = 1.0

    def _nearest_floor_cell(self, origin):
        y0, x0 = origin
        if not self._in_bounds((y0, x0)):
            y0 = int(np.clip(y0, 0, self.H - 1))
            x0 = int(np.clip(x0, 0, self.W - 1))
        visited = {(y0, x0)}
        queue = deque([(y0, x0)])
        while queue:
            y, x = queue.popleft()
            if self.building_map[y, x] != 1:
                return (y, x)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if self._in_bounds((ny, nx)) and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        return None

    def _propose_move(
        self,
        pos: Position,
        delta: Position,
        occupied: set[Position],
        reserved: set[Position],
    ) -> Tuple[Position, float]:
        y, x = pos
        dy, dx = delta
        nxt = (y + dy, x + dx)
        if not self._in_bounds(nxt):
            self.wall_collisions += 1
            return pos, -0.08
        if self.building_map[nxt] == 1:
            self.wall_collisions += 1
            return pos, -0.08
        if self.fire_intensity[nxt] > 0.05:
            # Active fire blocks movement the same way a wall does.
            # Counted under fire_blocked_moves (not wall_collisions) so
            # "ran into a wall" and "tried to walk into fire" stay
            # distinguishable in _info()/logging.
            self.fire_blocked_moves += 1
            return pos, -0.08
        if nxt in occupied and nxt != pos:
            self.agent_collisions += 1
            return pos, -0.05
        if nxt in reserved:
            self.agent_collisions += 1
            return pos, -0.05
        return nxt, 0.0

    def _has_line_of_sight(self, origin: Position, target: Position) -> bool:
        """Check for a clear straight line between `origin` and `target`.

        Marches cell-by-cell along the line using the same rounded-step DDA
        approach as observations.py's lidar_scan (round each intermediate
        (y, x) to the nearest grid cell), so this stays consistent with how
        the rest of the codebase already treats "is this direction clear."
        Unlike lidar_scan, this isn't restricted to 8 fixed compass angles
        or a max_range -- it checks the exact line to an arbitrary target
        cell, however far away and at whatever angle it actually is.

        Endpoints aren't checked: `origin` is always the agent's own
        (necessarily walkable) cell, and `target` is assumed to be a fire
        cell, which by construction never sits on a wall cell (fire_intensity
        is zeroed on wall cells -- see _fire_blocking_map). Only the cells
        strictly between them are checked for a wall blocking the line.

        Returns True if the line is clear (or origin == target), False if
        any intermediate cell is out of bounds or a wall.
        """
        y0, x0 = origin
        y1, x1 = target
        dy, dx = y1 - y0, x1 - x0
        steps = max(abs(dy), abs(dx))
        if steps == 0:
            return True
        for s in range(1, steps):
            t = s / steps
            yy = int(round(y0 + dy * t))
            xx = int(round(x0 + dx * t))
            if not self._in_bounds((yy, xx)) or self.building_map[yy, xx] == 1:
                return False
        return True

    def _nearest_normalized(self, pos: Position, targets: List[Position]) -> Tuple[float, float]:
        if not targets:
            return 0.0, 0.0
        y, x = pos
        target = min(targets, key=lambda p: abs(p[0] - y) + abs(p[1] - x))
        return target[0] / max(1, self.H - 1), target[1] / max(1, self.W - 1)

    def _bfs_distance_from_sources(self, sources: List[Position]) -> np.ndarray:
        """Multi-source BFS shortest-path distance (in grid steps) to the
        nearest cell in `sources`, over the SAME passability rule agent
        movement actually uses in `_propose_move` (a cell blocks movement
        iff building_map == 1 -- everything else, including doors/floor/
        outside-ring cells, is walkable).

        This is deliberately kept separate from the Manhattan-distance
        reward shaping and observation features currently in step()/
        _nearest_normalized() -- those still use straight-line distance.
        Swapping either of those over to this BFS distance is a follow-up
        change, not part of adding this helper.

        Returns an (H, W) float32 array. Unreachable cells (walled off
        from every source, or every source itself unreachable/invalid)
        are left as np.inf so callers can distinguish "far" from
        "no path exists" instead of silently treating them the same.
        One multi-source BFS from all active fire cells at once is O(H*W)
        total, versus running a separate single-source BFS per agent.
        """
        dist = np.full((self.H, self.W), np.inf, dtype=np.float32)
        frontier: deque = deque()
        for (y, x) in sources:
            if self._in_bounds((y, x)) and self.building_map[y, x] != 1 and dist[y, x] == np.inf:
                dist[y, x] = 0.0
                frontier.append((y, x))

        while frontier:
            y, x = frontier.popleft()
            d = dist[y, x]
            for dy, dx in self.ACTIONS.values():
                ny, nx = y + dy, x + dx
                if (
                    self._in_bounds((ny, nx))
                    and self.building_map[ny, nx] != 1
                    and dist[ny, nx] == np.inf
                ):
                    dist[ny, nx] = d + 1.0
                    frontier.append((ny, nx))

        return dist

    def _fire_blocking_map(self) -> np.ndarray:
        blocking = self.building_map.copy()
        blocking[self.building_map == 0] = 1
        return blocking

    def _in_bounds(self, pos: Position) -> bool:
        y, x = pos
        return 0 <= y < self.H and 0 <= x < self.W


if __name__ == "__main__":
    env = MultiAgentFireEnv(num_agents=4)
    obs, info = env.reset(seed=42)
    print("reset", obs.shape, info)
    for _ in range(5):
        obs, rewards, terminated, truncated, info = env.step(env.action_space.sample())
        print(info["step"], rewards, info["fire_remaining"])
        if terminated or truncated:
            break
    env.close()