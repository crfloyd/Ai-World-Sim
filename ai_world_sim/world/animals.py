"""Animal entity definitions.

Animals use scripted behavior in V0 — they do NOT run the neural policy.
They exist to provide food, danger, and ecological pressure, not realism.

TODO: V1+ — give animals their own lightweight learned policies.
TODO: V1+ — add reproduction, population dynamics, and biome territories.
TODO: V1+ — herbivore diet pressure on berry / grass resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnimalSpecies(Enum):
    RABBIT = "rabbit"
    DEER = "deer"
    WOLF = "wolf"


# Default HP per species — overridden by world config at spawn time.
_DEFAULT_HP: dict[AnimalSpecies, float] = {
    AnimalSpecies.RABBIT: 20.0,
    AnimalSpecies.DEER: 40.0,
    AnimalSpecies.WOLF: 80.0,
}

# Meat yield when the animal is killed.
_MEAT_YIELD: dict[AnimalSpecies, int] = {
    AnimalSpecies.RABBIT: 1,
    AnimalSpecies.DEER: 3,
    AnimalSpecies.WOLF: 0,  # wolves are not food
}


@dataclass
class Animal:
    """A single scripted animal instance."""

    id: int
    position: tuple[int, int]  # (row, col)
    species: AnimalSpecies
    hp: float
    alive: bool = True

    @property
    def is_prey(self) -> bool:
        return self.species in (AnimalSpecies.RABBIT, AnimalSpecies.DEER)

    @property
    def is_predator(self) -> bool:
        return self.species == AnimalSpecies.WOLF

    @property
    def meat_yield(self) -> int:
        return _MEAT_YIELD[self.species]

    @classmethod
    def create(
        cls,
        animal_id: int,
        position: tuple[int, int],
        species: AnimalSpecies,
        config: dict,
    ) -> "Animal":
        """Factory that reads HP from world config."""
        anim_cfg = config.get("animals", {})
        hp_key = f"{species.value}_hp"
        hp = float(anim_cfg.get(hp_key, _DEFAULT_HP[species]))
        return cls(id=animal_id, position=position, species=species, hp=hp)
