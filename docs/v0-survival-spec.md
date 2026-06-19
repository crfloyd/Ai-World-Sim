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

Only berries and water are consumable.

Trees and stone exist for future systems.

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
4 seasons
fixed days per season
```

Exact season lengths should be configurable.

---

## Agent Model

### Core State

Each agent contains:

```
hp
hunger
thirst
tired
inventory
position
home_position
```

All normalized values should be represented in the range `0.0` to `1.0` where appropriate.

---

### HP

```
0.0 = dead
1.0 = full health
```

HP decreases from:

- starvation
- dehydration
- predator attacks

HP regenerates slowly over time.

Sleeping accelerates healing.

---

### Hunger

```
0.0 = full
1.0 = starving
```

Increases every tick.

High hunger eventually damages HP.

---

### Thirst

```
0.0 = hydrated
1.0 = dehydrated
```

Increases every tick.

Should become dangerous faster than hunger.

---

### Tired

```
0.0 = rested
1.0 = exhausted
```

Increases every tick.

Movement and hunting accelerate tiredness.

High tiredness should reduce long-term survival.

---

### Home

Each agent has a home location.

Initially this can simply be the spawn location.

Home provides:

- safe sleeping
- food storage
- future expansion point

Sleeping at home is always safer than sleeping elsewhere.

---

### Animals

Animals use scripted behavior in V0.

Animals do not use neural policies.

Examples:

```
rabbit
deer
wolf
```

Rabbits and deer:

- wander
- eat
- flee

Wolves:

- wander
- hunt
- attack vulnerable agents

Animals exist to provide:

- food
- danger
- ecological pressure

---

## Observation Model

The neural policy never sees the full world.

It receives a local observation.

---

### Sight Radius

```
10 tiles

Observation window: 21 × 21

Agent remains centered.
```

---

### Observation Tensor

Use layered channels.

Channels:

```
terrain_grass
terrain_forest
terrain_mountain
terrain_water
soil_poor
soil_normal
soil_fertile
resource_berries
resource_tree
resource_stone
entity_agent
entity_animal
entity_predator
structure_home
out_of_bounds
```

Represented as:

```
channels × 21 × 21
```

---

### Self Features

Provide:

```
hp
hunger
thirst
tired
food_inventory
is_home
distance_to_home
season
day_progress
```

---

### Action Mask

Every observation contains an `action_mask`.

Invalid actions must be masked.

Examples:

- cannot drink if no water tile is adjacent
- cannot hunt if no animal target is in range
- cannot retrieve if food storage is empty

---

## Actions

V0 actions:

```
move_north
move_south
move_east
move_west
forage
hunt
drink
eat
rest
sleep
store_food
retrieve_food
```

Keep action count intentionally small.

---

### Rest

Rest is short-term recovery.

Effects:

- small tired reduction
- small healing
- maintains awareness

---

### Sleep

Sleep is long-term recovery.

Effects:

- large tired reduction
- faster healing
- reduced awareness

---

### Sleeping at Home

Benefits:

- safe
- fast recovery
- lower risk

---

### Sleeping Outside

Sleeping outside is dangerous.

While sleeping outside:

- reduced perception
- cannot flee effectively
- predators gain advantage

Sleeping outside should not guarantee death.

It should introduce meaningful risk.

---

## Memory System

Memory is external to the neural network.

The neural network receives memory features as part of its input.

Memory stores:

```
home_location
known_food_locations
known_water_locations
known_danger_locations
recent_success
recent_failure
```

The memory system is engineered.

Behavior is learned.

---

## Rewards

Rewards should encourage survival pressure.

Do not reward specific professions.

---

### Positive Rewards

Examples:

- surviving a day
- maintaining health
- reducing hunger
- reducing thirst
- storing food
- finding reliable food sources

---

### Negative Rewards

Examples:

- starvation
- dehydration
- injury
- excessive exhaustion
- death

---

### Important Rule

Never directly reward:

- farming
- hunting
- foraging

Only reward outcomes.

The policy should discover useful behaviors itself.

---

## Training Architecture

Training mode:

- many parallel world seeds
- short episodes
- minimal logging
- fast execution

---

### Story Mode

Story mode uses a frozen policy checkpoint.

No learning occurs.

The purpose is observation.

Story mode should:

- run longer
- record events
- produce summaries

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

Events should contain:

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
