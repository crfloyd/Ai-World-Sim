# AI World Sim

A simulation framework for training a single shared neural-network "person brain" inside procedurally generated RPG-style worlds.

The long-term objective is to train the policy across millions of generated worlds, freeze checkpoints at various stages, and run long observed simulations to study emergent behavior and generate stories.

---

## Core Idea

| Concept | Description |
|---|---|
| **One policy** | A single actor-critic neural network shared across all agents |
| **Many individuals** | Each agent is an instance of the same policy, differentiated by traits and context |
| **No hardcoded roles** | Farming, hunting, trading, crime, migration must emerge from learned behavior |
| **Emergent behavior** | Actions arise from the interaction of environment, traits, memory, inventory, skills, and local conditions |

This is **not** an LLM-agent project. It is **not** a behavior-tree project.

---

## Architecture

```
ai_world_sim/
├── world/              # Pure simulation — no RL coupling
│   ├── terrain.py      # TerrainType, SoilQuality, Cell dataclass
│   ├── entities.py     # Agent (id, position, hp, hunger, fatigue, inventory, traits, skills, memory)
│   ├── generator.py    # Seeded procedural world generation
│   ├── sim.py          # WorldSim — tick loop, agent actions, event dispatch
│   └── systems/
│       ├── seasons.py  # Season progression from day counter
│       ├── ecology.py  # Passive ecological processes (placeholder)
│       ├── health.py   # Hunger, fatigue, HP decay per tick
│       └── resources.py # Seasonal resource regeneration
│
├── rl/                 # RL coupling — depends on world/ but not vice-versa
│   ├── env.py          # Gymnasium environment (one world, one agent)
│   ├── observations.py # Flat obs builder: 5x5 grid window + agent state vector
│   ├── rewards.py      # Survival reward function
│   ├── model.py        # PyTorch actor-critic (CNN + MLP + shared trunk)
│   └── train.py        # RLlib PPO training entry point
│
├── story/
│   ├── events.py       # EventLog — timestamped action log per episode
│   └── summaries.py    # Episode summary generator
│
└── common/
    └── config.py       # YAML config loading / merging helpers
```

### Observation Space

The agent sees a **5x5 local grid window** (centered on itself) encoded into 5 channels:

| Channel | Content |
|---|---|
| 0 | Terrain type (normalised) |
| 1 | Soil quality (normalised) |
| 2 | Tree count (normalised) |
| 3 | Berry count (normalised) |
| 4 | Stone count (normalised) |

Plus an **11-dimensional state vector**:

| Index | Content |
|---|---|
| 0 | HP (normalised) |
| 1 | Hunger (normalised) |
| 2 | Fatigue (normalised) |
| 3-5 | Carried berries, wood, stone (log-normalised) |
| 6-9 | Season one-hot (Spring/Summer/Autumn/Winter) |
| 10 | Day within year (normalised) |

Total observation: **5x5x5 + 11 = 136 floats**.

### Action Space

| Index | Action |
|---|---|
| 0 | `move_north` |
| 1 | `move_south` |
| 2 | `move_east` |
| 3 | `move_west` |
| 4 | `forage` |
| 5 | `rest` |

Invalid actions (moving into water/mountain, foraging empty cell) are masked
to -inf before the softmax so the policy never wastes gradient on impossible choices.

### Policy Model

```
flat_obs (136,)
    |
    +--- grid portion ---> reshape (5, 5x5) ---> Conv2d(16) ---> Conv2d(32) ---> flatten ---> cnn_embed
    |
    +--- state portion ---> Linear(128) ---> Linear(64) ---> state_embed
                                                                          |
                                                      concat(cnn_embed + state_embed)
                                                                          |
                                                           Linear(256) ---> Linear(128)   <- shared trunk
                                                          /                     \
                                                  policy_head              value_head
                                                (logits, 6)                 (scalar)
```

---

## Training Flow

```
configs/train.yaml          <- hyperparameters, env config
configs/world.yaml          <- world size, resource density, agent stats
        |
        v
WorldEnv.reset(seed=random) <- new world per episode from seed pool
        |
        +-- WorldSim.generate()
        +-- spawn_agents(1)
        +-- returns obs dict {"obs": ..., "action_mask": ...}
        |
        v
PPO worker loop
  +-- collect rollouts (num_rollout_workers x num_envs_per_worker envs in parallel)
  +-- compute GAE advantages
  +-- SGD on shared AgentBrain weights
        |
        v
checkpoint saved every N iterations
```

---

## Terrain & World

| Type | Passable | Resources |
|---|---|---|
| Grass | Yes | Berries, some trees, stone |
| Forest | Yes | Trees, some berries, stone |
| Mountain | No | - |
| Water | No | - |

**Seasons** cycle Spring -> Summer -> Autumn -> Winter every 30 days (configurable).
Resource regeneration rate drops in Autumn and stops in Winter.

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt && pip install -e .
```

### Run tests

```bash
pytest tests/ -v
```

### Run a quick sanity check (no RL deps required)

```python
from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.story.events import EventLog
from ai_world_sim.story.summaries import episode_summary

cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
sim = WorldSim(config=cfg, seed=42)
sim.generate()
log = EventLog()
sim.attach_event_log(log)
agents = sim.spawn_agents(1)
agent = agents[0]

for _ in range(sim.ticks_per_day * 10):   # simulate 10 days
    sim.forage(agent)
    sim.tick()

print(episode_summary(sim, agent, log))
```

### Start training

```bash
python -m ai_world_sim.rl.train
# or with a custom config:
python -m ai_world_sim.rl.train --config configs/train.yaml
```

Training logs iteration number, mean episode reward, and mean episode length.
Checkpoints are saved to `checkpoints/` every 50 iterations by default.

---

## Configuration

All knobs live in `configs/`. No magic numbers in code.

**`configs/world.yaml`** — world size, terrain thresholds, resource density, agent base stats, season length.

**`configs/train.yaml`** — PPO hyperparameters, number of workers, episode length, checkpoint frequency, model architecture.

---

## Future Roadmap

### Near term
- [ ] Multi-agent support (N agents per world, shared policy via RLlib MARL)
- [ ] Eating from inventory to reduce hunger
- [ ] Visible agent-to-agent interaction (proximity detection in observation)
- [ ] Simple crafting actions (wood + stone -> tool)

### Medium term
- [ ] Animal entities (prey, predators) with their own policies
- [ ] Trade system — agents exchange items
- [ ] Trust / reputation scores between agents
- [ ] Long-term episodic memory buffer exposed to the policy

### Long term
- [ ] Crime and conflict (theft, assault, territorial behavior)
- [ ] Faction / settlement emergence
- [ ] Migration pressure driven by seasons and resource depletion
- [ ] Story generation: pipe event logs to a language model narrator
- [ ] Cross-seed statistical analysis of emergent behavioral patterns
- [ ] Curriculum learning: start with abundant worlds, increase scarcity

---

## Design Principles

1. **World simulation is independent from RL.** `ai_world_sim/world/` has zero RLlib or Gymnasium imports.
2. **Configuration-driven.** Every tunable constant lives in `configs/`.
3. **No hardcoded roles.** No label "farmer", "hunter", "guard" anywhere in the simulation.
4. **Extensibility over optimization.** Clarity first; profile before optimizing.
5. **Type-hinted throughout.** Every public function has full type annotations.
