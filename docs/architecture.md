# AI World Sim — Architecture

## Core Loop

The backbone of the entire system is a single closed loop that repeats once per tick:

```
┌──────────────┐
│   WorldSim   │◄─────────────────────────────────────────┐
│              │                                           │
│  grid        │                                           │
│  agents      │                                           │
│  animals     │                                           │
│  season/day  │                                           │
└──────┬───────┘                                           │
       │                                                   │
       │ raw world state                                   │ action applied
       ▼                                                   │
┌──────────────────────┐                                   │
│  Observation Builder │                                   │
│  (rl/observations.py)│                                   │
│                      │                                   │
│  build_grid_tensor() │─► local_grid   (15, 21, 21)      │
│  build_self_features()─► self_features (14,)             │
│  build_action_mask() │─► action_mask  (12,)              │
└──────────────────────┘                                   │
       │                                                   │
       │ also calls                                        │
       ▼                                                   │
┌──────────────────────┐                                   │
│   Memory Summarizer  │                                   │
│   (world/memory.py)  │                                   │
│                      │                                   │
│  .update()  ◄────── scans visible area                  │
│  .decay()   ◄────── ages existing entries               │
│  .summarize()──────► memory_features (12,)               │
└──────────────────────┘                                   │
       │                                                   │
       │ {local_grid, self_features,                       │
       │  memory_features, action_mask}                    │
       ▼                                                   │
┌──────────────────────────────────────────────────────┐   │
│                    Policy Network                     │   │
│                   (rl/model.py)                      │   │
│                                                      │   │
│  local_grid ──► CNN ──────────────────► cnn_embed   │   │
│                 (Conv×3 + GlobalAvgPool)              │   │
│                                                      │   │
│  self_features ──► MLP ─────────────► self_embed    │   │
│                                                      │   │
│  memory_features ──► MLP ──────────► memory_embed   │   │
│                                                      │   │
│  concat[cnn_embed, self_embed, memory_embed]         │   │
│          │                                           │   │
│          └──► Shared Trunk (MLP) ──► trunk_out      │   │
│                                      │               │   │
│                             ┌────────┴────────┐      │   │
│                             ▼                 ▼      │   │
│                        policy_head       value_head  │   │
│                        (logits, 12)      (scalar)    │   │
│                             │                        │   │
│                      action_mask applied             │   │
│                      (-1e9 for invalid)              │   │
└──────────────────────────────┬───────────────────────┘   │
                               │                           │
                               │ action (int, 0–11)        │
                               ▼                           │
                    ┌──────────────────┐                   │
                    │  WorldSim action │───────────────────┘
                    │  dispatch        │
                    │                  │
                    │  move / forage   │
                    │  hunt / drink    │
                    │  eat / rest      │
                    │  sleep / store   │
                    └──────────────────┘
```

---

## Observation Breakdown

### local_grid — (15, 21, 21)

A 21×21 window centred on the agent (sight radius = 10 tiles).
Each cell is encoded as 15 binary/normalised channels:

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
  7  resource_berries   berries / max_berries   ∈ [0, 1]
  8  resource_trees     trees / max_trees       ∈ [0, 1]
  9  resource_stone     stone / max_stone       ∈ [0, 1]
 10  entity_agent       1.0 if other agent here (future)
 11  entity_prey        1.0 if rabbit or deer here
 12  entity_predator    1.0 if wolf here
 13  structure_home     1.0 if this is agent's home tile
 14  out_of_bounds      1.0 if outside world boundary
```

Adding a new channel = add a constant in terrain.py + one line in build_grid_tensor().

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

Memory entries decay at `decay_rate` confidence per tick and are pruned at 0.

---

### action_mask — (12,)

Binary float32. 1.0 = valid, 0.0 = blocked.
Invalid actions get logit − 1e9 so they have ~0 probability after softmax.

```
 Idx  Action         Blocked when
 ───  ─────────      ─────────────────────────────────────────────
  0   move_north     target cell is impassable or out of bounds
  1   move_south
  2   move_east
  3   move_west
  4   forage         current cell has no berries/trees/stone
  5   hunt           no living prey in adjacent cell
  6   drink          no water tile adjacent (cardinal only)
  7   eat            inventory has no berries or meat
  8   rest           always valid
  9   sleep          always valid
 10   store_food     not at home, OR inventory has no food
 11   retrieve_food  not at home, OR stored_food is empty
```

---

## Policy Network

```
Input                    Encoder                 Output dim
────────────────────     ───────────────────     ──────────
(B, 15, 21, 21)    ──►  Conv(32) → Conv(64)  ─► (B, 64)
local_grid               → Conv(64) → GlobAvgPool
                         (spatial context)

(B, 14)            ──►  Linear(128) → ReLU   ─► (B, 64)
self_features            → Linear(64)
                         (vital stats + location)

(B, 12)            ──►  Linear(64) → ReLU    ─► (B, 32)
memory_features          → Linear(32)
                         (known world locations)

concat: (B, 64+64+32) = (B, 160)
           │
           ▼
     Linear(256) → ReLU → Linear(128) → ReLU   (shared trunk)
           │                     │
           ▼                     ▼
     policy_head           value_head
     Linear(128,12)        Linear(128,1)
     + action mask         scalar V(s)
           │
           ▼
     Categorical(logits)
           │
         action
```

---

## World Simulation Layer

```
WorldSim
├── grid: list[list[Cell]]          terrain + soil + resources per cell
├── agents: list[Agent]             neural policy instances
├── animals: list[Animal]           scripted behavior (rabbit/deer/wolf)
│
├── systems/
│   ├── SeasonSystem                day → season lookup
│   ├── HealthSystem                hunger/thirst/tired decay + action effects
│   ├── ResourceSystem              seasonal berry/tree/stone regen
│   ├── AnimalSystem                scripted wolf/prey behavior per tick
│   └── EcologySystem               placeholder (V1+: animals, soil, fire)
│
└── tick() order:
    1. HealthSystem.tick()          per-agent stat decay + death check
    2. AnimalSystem.tick()          wolf attacks, prey fleeing
    3. ResourceSystem.tick()        regen based on season
    4. EcologySystem.tick()         (noop v0.1)
    5. clock advance → season check
```

---

## File Map

```
ai_world_sim/
├── world/                    # Pure simulation — zero RL imports
│   ├── terrain.py            # TerrainType, SoilQuality, Cell, channel constants
│   ├── entities.py           # Agent (all state, home, stored_food, MemoryStore ref)
│   ├── animals.py            # Animal, AnimalSpecies, meat yield
│   ├── memory.py             # MemoryStore, MemoryEntry, MEMORY_DIM
│   ├── generator.py          # Seeded procedural world + animal spawn
│   ├── sim.py                # WorldSim: generate, tick, all actions
│   └── systems/
│       ├── seasons.py        # SeasonSystem
│       ├── health.py         # HealthSystem (hunger/thirst/tired/sleep)
│       ├── resources.py      # ResourceSystem (seasonal regen)
│       ├── animals.py        # AnimalSystem (scripted behavior)
│       └── ecology.py        # EcologySystem (placeholder)
│
├── rl/                       # RL coupling — depends on world/, not vice-versa
│   ├── observations.py       # build_observation(), build_action_mask(), constants
│   ├── rewards.py            # survival_reward()
│   ├── env.py                # WorldEnv (Gymnasium, 12-action Dict obs)
│   ├── model.py              # AgentBrain (CNN + dual MLP + trunk + heads)
│   └── train.py              # RLlib PPO, ActionMaskModel wrapper
│
├── story/
│   ├── events.py             # EventLog, Event
│   └── summaries.py          # episode_summary(), season_report()
│
└── common/
    └── config.py             # load_config(), merge_configs(), default paths
```

---

## Training vs. Story Mode

```
TRAINING MODE                          STORY MODE
─────────────────────────────          ─────────────────────────────
Many parallel envs                     Single env
Random seed per episode                Fixed story seed
Short episodes (1000 steps)           Long episodes (days/seasons)
Gradient updates every batch           No gradient updates
Minimal logging                        Full EventLog capture
Fast execution                         Episode summary generated
```

---

## Seed Pools

Defined in `configs/world.yaml`:

```
seeds.training_range   [0, 899999]   — sampled randomly per episode
seeds.validation       10 fixed      — used for evaluation callbacks (not training)
seeds.story            5 fixed       — reserved for observed narrative runs
```

The story seeds are never used during training so the policy has never been
optimised against those specific worlds.
