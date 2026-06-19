# AI World Sim — V0 Survival Specification

## Goal

The purpose of V0 is to prove that a single shared neural policy can learn basic survival behaviors inside a deterministic procedurally generated world.

This version is intentionally limited.

There is:

- no farming
- no trading
- no factions
- no crime
- no social systems
- no relationships
- no crafting

The only objective is survival.

The policy should eventually learn behaviors such as:

- finding food
- finding water
- hunting animals
- sleeping safely
- storing food
- returning home
- avoiding death

The purpose of V0 is to validate:

- world simulation
- observation encoding
- action encoding
- reward design
- PPO training loop
- memory architecture
- event logging

---

## Core Design Principles

### Shared Brain

There is only one neural policy.

There are not separate brains for:

- farmer
- hunter
- trader
- guard

Every agent uses the same policy network.

Behavior differences emerge from:

- environment
- inventory
- skills
- memory
- traits
- location
- experience

The world teaches the policy how to survive.

---

### Deterministic Simulation

The simulator must be deterministic.

Given:

```
world_seed
agent_seed
config
policy_checkpoint
```

the simulation must replay identically.

All randomness must originate from seeded generators.

---

## World Model

### World Type

Grid-based world.

Each tile contains:

```
terrain
soil
resource
structure
occupant
```

Only one occupant may exist per tile.

**Default world size:** 64 × 64 tiles (configurable in `configs/world.yaml`).

---

### Terrain Types

```
grass
forest
mountain
water
```

Terrain affects:

- movement cost
- visibility (future)
- resource spawning

Mountain and water tiles are impassable.

---

### Soil Types

```
poor
normal
fertile
```

Not used in V0.

Included now because farming will depend on it later.

---

### Resources

V0 resources:

```
berries
trees
stone
water
```

Only berries and water are consumable in V0.

Trees and stone exist for future systems.

**Regeneration rates (per tick):**

```
spring:  0.005
summer:  0.003
autumn:  0.002
winter:  0.000   (no regen)
```

---

### Seasons

The world has:

```
spring
summer
autumn
winter
```

Each season influences:

- berry regeneration
- animal population
- survival difficulty

Winter should be noticeably harder than summer.

---

### Time

```
96 ticks per day
30 days per season   (configurable)
4 seasons per year
```

---

## Agent Model

### Core State

Each agent contains:

```
hp           — current health
hunger       — 0.0 (full) to 100.0 (starving)
thirst       — 0.0 (hydrated) to 100.0 (dehydrated)
tired        — 0.0 (rested) to 100.0 (exhausted)
inventory    — dict of carried items (berries, meat)
stored_food  — dict of items stashed at home
position     — (row, col)
home_position— (row, col)  fixed at spawn time
sleeping     — bool flag
```

---

### HP

```
0.0 = dead
100.0 = full health
```

HP decreases from:

- starvation (`-2.0 hp/tick` when hunger = 100)
- dehydration (`-3.0 hp/tick` when thirst = 100)
- predator attacks

Sleeping accelerates healing.

---

### Hunger

```
0.0 = full
100.0 = starving
```

Increases by `0.3` per tick.

Starvation damage begins when hunger reaches maximum.

---

### Thirst

```
0.0 = hydrated
100.0 = dehydrated
```

Increases by `0.5` per tick — faster than hunger.

Dehydration damage (`-3.0 hp/tick`) begins when thirst reaches maximum.

---

### Tired

```
0.0 = rested
100.0 = exhausted
```

Increases by `0.2` per tick.

Movement and hunting each cost an additional `0.15` tired per action.

---

### Home

Each agent has a home location set to its spawn position.

Home provides:

- safe sleeping (wolves never attack agents sleeping at home)
- food storage
- future expansion point

Sleeping at home gives bonus tired recovery and HP recovery on top of the base sleep effect.

---

### Animals

Animals use scripted behavior in V0.

Animals do not use neural policies.

Species:

```
rabbit   — hp 20   meat yield 1
deer     — hp 40   meat yield 3
wolf     — hp 80   no meat yield
```

Rabbits and deer:

- wander randomly
- flee from wolves and agents within `flee_range` (5 tiles)

Wolves:

- hunt agents and prey within `hunt_range` (10 tiles)
- attack adjacent agents (`-15.0 hp/attack`; `+10.0 bonus` vs sleeping agents)
- wolves never attack agents sleeping at their home position

Animals exist to provide:

- food (hunting prey)
- danger (wolf attacks)
- ecological pressure

**Default population:** 20 rabbits, 10 deer, 3 wolves per world.

---

## Observation Model

The neural policy never sees the full world.

It receives a local observation dict with four keys:

```
local_grid      — spatial grid around the agent
self_features   — agent's own vital stats and context
memory_features — compact summary of remembered locations
action_mask     — binary validity mask over all 12 actions
```

---

### Sight Radius

```
sight_radius = 10 tiles
observation window = 21 × 21
agent remains centered
```

---

### local_grid — (15, 21, 21)

A 21×21 window centred on the agent.
Each cell is encoded as 15 binary or normalised channels:

```
 Ch  Name               Encoding
 ──  ─────────────      ──────────────────────────────────────
  0  terrain_grass      1.0 if terrain == GRASS
  1  terrain_forest     1.0 if terrain == FOREST
  2  terrain_mountain   1.0 if terrain == MOUNTAIN
  3  terrain_water      1.0 if terrain == WATER
  4  soil_poor          1.0 if soil == POOR
  5  soil_normal        1.0 if soil == NORMAL
  6  soil_fertile       1.0 if soil == FERTILE
  7  resource_berries   berries / max_berries  ∈ [0, 1]
  8  resource_trees     trees / max_trees      ∈ [0, 1]
  9  resource_stone     stone / max_stone      ∈ [0, 1]
 10  entity_agent       1.0 if another agent is here (V1+)
 11  entity_prey        1.0 if rabbit or deer here
 12  entity_predator    1.0 if wolf here
 13  structure_home     1.0 if this is the agent's home tile
 14  out_of_bounds      1.0 if outside world boundary
```

Channel constants are defined in `ai_world_sim/world/terrain.py`.

---

### self_features — (14,)

```
 Idx  Feature              Encoding
 ───  ─────────────────    ─────────────────────────────────────
  0   hp                   hp / max_hp                ∈ [0, 1]
  1   hunger               hunger / max_hunger        ∈ [0, 1]
  2   thirst               thirst / max_thirst        ∈ [0, 1]
  3   tired                tired / max_tired          ∈ [0, 1]
  4   berries_carried      log1p(n) / log1p(20)       ∈ [0, 1]
  5   meat_carried         log1p(n) / log1p(20)       ∈ [0, 1]
  6   stored_food_total    log1p(n) / log1p(50)       ∈ [0, 1]
  7   is_at_home           1.0 / 0.0
  8   dist_to_home         euclidean / world_diagonal ∈ [0, 1]
  9   season_spring        one-hot
 10   season_summer        one-hot
 11   season_autumn        one-hot
 12   season_winter        one-hot
 13   day_progress         day_in_year / year_length  ∈ [0, 1]
```

---

### memory_features — (12,)

```
 Idx  Feature                  Encoding
 ───  ──────────────────────── ──────────────────────────────────
  0   nearest_food_dist        manhattan / world_diagonal
  1   nearest_food_dir_r       normalised row direction ∈ [-1, 1]
  2   nearest_food_dir_c       normalised col direction ∈ [-1, 1]
  3   nearest_water_dist       manhattan / world_diagonal
  4   nearest_water_dir_r
  5   nearest_water_dir_c
  6   nearest_danger_dist      manhattan / world_diagonal
  7   nearest_danger_dir_r
  8   nearest_danger_dir_c
  9   num_known_food           log1p(n) / log1p(max_entries)
 10   num_known_water
 11   num_known_danger
```

Memory entries are updated each time an observation is built (the agent scans its visible area), then decay at `0.02` confidence per tick and are pruned when confidence reaches 0.

Home location is not in the memory features — it is always in `self_features` as `dist_to_home` and is always known.

---

### action_mask — (12,)

Binary float32. `1.0` = valid, `0.0` = blocked.

Invalid actions get logit `−1e9` before softmax so they have ~0 probability.

```
 Idx  Action         Blocked when
 ───  ─────────      ─────────────────────────────────────────────
  0   move_north     target cell is impassable or out of bounds
  1   move_south
  2   move_east
  3   move_west
  4   forage         current cell has no berries, trees, or stone
  5   hunt           no living prey in adjacent cell (cardinal)
  6   drink          no water tile adjacent (cardinal only)
  7   eat            inventory has no berries or meat
  8   rest           always valid
  9   sleep          always valid
 10   store_food     not at home, OR inventory has no food
 11   retrieve_food  not at home, OR stored_food is empty
```

At least one action is always valid — `rest` is the unconditional fallback.

---

## Actions

12 discrete actions:

```
 Idx  Action
 ───  ─────────────
  0   move_north
  1   move_south
  2   move_east
  3   move_west
  4   forage
  5   hunt
  6   drink
  7   eat
  8   rest
  9   sleep
 10   store_food
 11   retrieve_food
```

---

### Forage

Adds items to inventory (does not directly reduce hunger).

Agents must use the `eat` action to consume carried food.

```
berries_gained = forage_berry_amount (default 2)
```

---

### Hunt

Kills adjacent prey and adds meat to inventory.

Costs `hunt_base_hp_cost` (2.0) hp regardless of success.

Only valid when living prey is adjacent (cardinal directions).

---

### Drink

Reduces thirst by `drink_thirst_reduction` (40.0).

Only valid when a water tile is adjacent (cardinal directions).

---

### Eat

Consumes berries or meat from inventory.

```
berries:  hunger -= eat_hunger_reduction       (25.0)
meat:     hunger -= eat_meat_hunger_reduction  (40.0)
```

Meat eaten first if both are available.

---

### Rest

Short-term recovery.

Effects:

- `tired -= rest_tired_recovery` (5.0)
- `hp += rest_hp_recovery` (0.5)
- sleeping flag cleared

---

### Sleep

Long-term recovery.

Effects:

- `tired -= sleep_tired_recovery` (20.0)
- `hp += sleep_hp_recovery` (2.0)
- sleeping flag set to `True`

**At home bonus:**

- additional `tired -= sleep_home_tired_bonus` (10.0)
- additional `hp += sleep_home_hp_bonus` (1.0)

---

### Sleeping Outside

While `sleeping = True` and not at home:

- wolves deal `+wolf_sleep_bonus_damage` (10.0) extra damage on attack
- the agent cannot take defensive action

Sleeping outside should not guarantee death. It introduces meaningful risk that the policy must learn to manage.

---

### Store Food / Retrieve Food

Transfer food between inventory and home storage.

Both actions require the agent to be at `home_position`.

`store_food` moves all carried berries and meat to `stored_food`.

`retrieve_food` moves all stored food back to inventory.

---

## Memory System

Memory is an external engineered system — it is not inside the neural network.

The neural policy receives a compact 12-dimensional `memory_features` vector derived from the memory store.

The memory store tracks three categories:

```
known_food_locations    — cells with observed berries or trees
known_water_locations   — cells with water terrain
known_danger_locations  — last observed positions of wolves
```

Each entry stores:

```
position         — (row, col)
confidence       — 1.0 when observed; decays at 0.02/tick; pruned at 0
last_seen_tick   — tick at which the entry was last confirmed
```

The store is capped at `max_entries` (default 20) per category. When full, the least-confident entry is replaced.

Memory is updated every time `build_observation()` is called — the agent scans its entire visible area and upserts any food, water, or danger it can see.

The behavior of where to go is learned; the mechanics of remembering are engineered.

---

## Rewards

Rewards encourage survival pressure without rewarding specific behaviors.

**Rule:** Never directly reward farming, hunting, foraging, or specific action choices. Only reward outcomes.

```
+0.01   per tick alive                        (ALIVE_REWARD)
-0.001 × hunger                               (HUNGER_PENALTY_SCALE)
-0.0015 × thirst                              (THIRST_PENALTY_SCALE; higher than hunger)
-0.0005 × tired                               (TIRED_PENALTY_SCALE)
+0.05   on successfully storing food          (shaping: rewards planning ahead)
-1.0    on death                              (DEATH_PENALTY)
```

The penalty scales are applied to the raw stat values (0–100), so a fully hungry agent
loses `0.1` per tick from hunger alone, while a fresh agent earns net `+0.01`.

The `store_food` shaping bonus is a temporary training aid and should be removed once the policy is stable.

---

## Training Architecture

Training mode:

- many parallel world seeds (range 0–899,999)
- short episodes (1,000 steps by default)
- minimal logging
- fast execution

---

### Validation

10 fixed seeds (900001–900010) reserved for evaluation callbacks.

These seeds are never used during training so evaluation measures true generalization.

---

### Story Mode

Story mode uses a frozen policy checkpoint.

No learning occurs.

The purpose is observation.

Story mode should:

- run longer
- record events
- produce summaries

5 fixed story seeds (999001–999005) are reserved and never used during training.

---

## Event Logging

Every meaningful event should be logged.

Examples:

```
Agent 4 moved north
Agent 9 drank water
Agent 12 hunted deer
Agent 5 stored food
Agent 7 slept outside
Agent 3 died
Season changed to winter
```

Events contain:

```
tick
day
season
agent_id
position
event_type
metadata
```

These logs will eventually become the basis for emergent narrative generation.

---

## Policy Network

The shared neural policy is a CNN + dual-MLP actor-critic:

```
local_grid (15, 21, 21)  ──►  Conv×3 + GlobalAvgPool  ──► (64,)
self_features (14,)      ──►  MLP [128, 64]            ──► (64,)
memory_features (12,)    ──►  MLP [64, 32]             ──► (32,)

concat → (160,)
  │
  └──► shared trunk MLP [256, 128]
            │
     ┌──────┴──────┐
     ▼             ▼
policy_head    value_head
Linear(128,12) Linear(128,1)
+ action mask  scalar V(s)
```

Global average pooling keeps the CNN output size fixed at `(B, 64)` regardless of window size.

---

## Success Criteria

V0 is successful when a trained policy can reliably:

- find food
- find water
- avoid starvation
- avoid dehydration
- sleep safely
- return home
- store food
- survive multiple seasons

without hardcoded survival behavior.

The neural policy should learn survival from environmental pressures rather than scripted decision logic.

---

## Future Systems (Out of Scope for V0)

The following are V1+ features. None of these should be designed, stubbed, or partially implemented in V0.

| System | Status |
|---|---|
| Farming | V1+ |
| Crafting | V1+ |
| Trade | V1+ |
| Crime | V1+ |
| Guards | V1+ |
| Factions | V1+ |
| Trust | V1+ |
| Reputation | V1+ |
| Social memory | V1+ |
| Ownership | V1+ |
| Economy | V1+ |
| Settlements | V1+ |
| Warfare | V1+ |
| Leadership | V1+ |
| Advanced animal ecosystems | V1+ |
| Weather | V1+ |
| Disease | V1+ |
| Genetic traits | V1+ |
| Lifelong learning | V1+ |
| Multi-agent coordination | V1+ |
| Neural memory (LSTM / attention) | V1+ |
