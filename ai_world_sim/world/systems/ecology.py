"""Ecological balance and passive world processes.

This module is intentionally sparse in v0.1 — it is the hook for future
emergent ecosystem mechanics such as:

TODO: Animal population dynamics (prey/predator cycles).
TODO: Soil fertility changes driven by farming or overuse.
TODO: Spread of forest across fertile grass cells over many seasons.
TODO: Disease or blight events that reduce resource caps.
TODO: Fire propagation (sparked by lightning or agents).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_world_sim.world.sim import WorldSim


class EcologySystem:
    """Placeholder for passive ecological processes that run each tick."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def tick(self, world: "WorldSim") -> None:
        """Apply one tick of ecological simulation.

        Currently a no-op; future sub-systems will be called here.
        """
        # TODO: animal population tick
        # TODO: soil fertility drift
        # TODO: forest spread (once per season)
        pass
