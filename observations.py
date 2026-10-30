"""Observation helper utilities for MultiAgentFireEnv
"""
import numpy as np


def lidar_scan(env, agent_id, num_rays: int = 8, max_range: int = 8) -> np.ndarray:
    """Cast `num_rays` outward from an agent and return normalized distance
    to the nearest wall or map boundary along each ray.

    Rays are spaced evenly around the full circle (0 = facing +x/"east"),
    walked one grid cell at a time via a simple DDA-style march. This gives
    the agent an explicit sense of how much open space surrounds it in
    each direction, independent of its narrow local_observation window --
    useful for learning to avoid dead ends and to steer toward open,
    unexplored space rather than hugging spawn corners.

    Returns:
        np.ndarray of shape (num_rays,), each value in [0, 1]:
            1.0 -> no obstruction found within max_range cells
            0.0 -> wall/boundary immediately adjacent
    """
    y0, x0 = env.agent_positions[agent_id]
    H, W = env.H, env.W
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    readings = np.ones(num_rays, dtype=np.float32)

    for i, theta in enumerate(angles):
        dy = np.sin(theta)
        dx = np.cos(theta)
        dist = max_range
        for step in range(1, max_range + 1):
            yy = int(round(y0 + dy * step))
            xx = int(round(x0 + dx * step))
            out_of_bounds = yy < 0 or yy >= H or xx < 0 or xx >= W
            hit_wall = (not out_of_bounds) and env.building_map[yy, xx] == 1
            if out_of_bounds or hit_wall:
                dist = step - 1
                break
        readings[i] = dist / max_range

    return readings.astype(np.float32)


def local_observation(env, agent_id, radius=2):
    """Return flattened local observation for an agent.

    Channels: wall (1), door (1 if building_map==2), fire_intensity, firefighter_presence
    Flattened to vector.
    """
    y0, x0 = env.agent_positions[agent_id]
    H, W = env.H, env.W
    size = 2 * radius + 1
    channels = 4
    local = np.zeros((channels, size, size), dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = y0 + dy
            xx = x0 + dx
            ly = dy + radius
            lx = dx + radius
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                local[0, ly, lx] = 1.0
                continue
            cell = int(env.building_map[yy, xx])
            local[0, ly, lx] = 1.0 if cell == 1 else 0.0
            local[1, ly, lx] = 1.0 if cell == 2 else 0.0
            if hasattr(env, 'fire_intensity') and env.fire_intensity is not None:
                local[2, ly, lx] = float(np.clip(env.fire_intensity[yy, xx], 0.0, 1.0))
            else:
                local[2, ly, lx] = 0.0
            local[3, ly, lx] = 1.0 if (yy, xx) in env.agent_positions else 0.0

    return local.reshape(-1)