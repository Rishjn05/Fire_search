"""Fire dynamics helper module
Provides simple, modular fire diffusion/spread/decay functionality.
This is intentionally lightweight and deterministic to be safe for tests.
"""

import numpy as np


def step_fire(fire_intensity, building_map, spread_rate=0.05, decay_rate=0.01, diffusion_rate=0.1):
    """Advance fire intensity one simulation step.

    Args:
        fire_intensity (np.ndarray): HxW float array (0..1).
        building_map (np.ndarray): HxW int array where 1=wall, other values walkable.
        spread_rate (float): base probability to increase neighbor intensity.
        decay_rate (float): fraction intensity decayed per step.
        diffusion_rate (float): weight for Laplacian diffusion.

    Returns:
        new_fire_intensity (np.ndarray): updated intensity clipped to [0,1].
    """
    H, W = fire_intensity.shape
    # Walls should act as a reflecting (no-flux) boundary, not an
    # absorbing sink. The naive Laplacian below always assumes 4
    # walkable neighbors, so a fire cell next to walls (whose intensity
    # is force-zeroed elsewhere) sees an artificial steep negative
    # gradient and "leaks" intensity into the wall every step -- this
    # happens regardless of decay_rate, and is most visible in corners
    # and dead-ends where fire can vanish even with decay_rate=0.
    free = (building_map != 1).astype(np.float32)
    fire_masked = fire_intensity * free

    neighbor_sum = (
        np.roll(fire_masked, 1, axis=0)
        + np.roll(fire_masked, -1, axis=0)
        + np.roll(fire_masked, 1, axis=1)
        + np.roll(fire_masked, -1, axis=1)
    )
    neighbor_count = (
        np.roll(free, 1, axis=0)
        + np.roll(free, -1, axis=0)
        + np.roll(free, 1, axis=1)
        + np.roll(free, -1, axis=1)
    )

    # Discrete Laplacian using only walkable neighbors (reflecting
    # boundary at walls: diffusion only happens between floor cells).
    lap = neighbor_sum - neighbor_count * fire_intensity
    diffusion = diffusion_rate * lap

    # Spread: neighbors gain a fraction proportional to spread_rate and
    # neighbor intensity. Wall neighbors contribute 0 (fire_masked is 0
    # there), so this term was already correct.
    neighbor_flux = neighbor_sum * (spread_rate * 0.25)

    mask = free
    new_int = fire_intensity + (diffusion + neighbor_flux) * mask

    # decay
    new_int = new_int * (1.0 - decay_rate)

    # clip
    new_int = np.clip(new_int, 0.0, 1.0)

    return new_int