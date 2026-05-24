# Nanocolony Lab — Architecture

## Overview

**Nanocolony Lab** is an artificial-life sandbox where thousands of autonomous agents compete for resources, emit signals, reproduce, and die in a 2D grid world. The goal is to study emergent behaviors like swarming, trail formation, and collective dynamics without centralized control.

## Core Principles

- **Local intelligence only** (no global view)
- **Simple rules → complex outcomes**
- **Sandboxed environment**
- **Measure emergence, don't assume it**
- **Iterate fast, observe everything**

## System Architecture

```
nanocolony/
├── main.py                # Entry point (interactive pygame)
├── config.py              # Global parameters
├── experiment.py          # Headless experiment runner
├── experiments.yml        # Experiment definitions
│
├── world/
│   └── environment.py     # 512x512 grid, resources, signal fields
│
├── agents/
│   ├── agent_state.py     # Vectorized agent data (positions, energy, traits)
│   └── agent_logic.py     # Sensing and rule-based decision logic
│
├── simulation/
│   └── step.py            # Main simulation update loop
│
├── rendering/
│   └── renderer.py        # Pygame visualization
│
├── analysis/
│   └── metrics.py         # Data tracking (currently unused)
│
└── experiments/
    └── configs/           # Experiment presets
```

## Simulation Loop

```python
for step in range(num_steps):
    # 1. Update: decay + diffuse signals, spawn & decay resources
    env.update()

    # 2. Sensor sampling (vectorized across all alive agents)
    inputs, alive = agent_logic.sense(agent_state)

    # 3. Decision (vectorized numpy, no per-agent Python loop)
    actions = agent_logic.compute_actions(inputs, agent_state, alive)

    # 4. Apply movement, turns, and existence cost
    agent_state.apply_actions(actions, alive)

    # 5. Consume resources, emit signals (food trails, depletion markers)
    reproducers = agent_logic.process_actions(agent_state, actions, alive)

    # 6. Reproduce and age/death
    agent_state.reproduce(reproducers)
    agent_state.age_step()
    dead_pos = agent_state.remove_dead()   # emit signal B at death sites

    # 7. Logging + compact dead-agent slots
    record_metrics()
    if step % 200 == 0:
        agent_state.compact()
```

## Signal Fields (Key for Emergence)

Two grid-wide fields (`signal_a`, `signal_b`) diffuse, decay, and are accumulated by agent/world events:

```python
field  *= SIGNAL_DECAY              # 0.97 = ~33 steps to 10%
field   = diffuse(field, rate=0.15) # 4-neighbor kernel
field  += emission                   # per-cell additive
```

### Signal A — Food Trail
| Trigger | Strength | When |
|---------|----------|------|
| Agent eats (consumed > 0.05) | `0.3 × trait_signal_a_strength` | Automatic, each step |
| Agent chooses ACTION_EMIT_A | `0.3 × trait_signal_a_strength` | Deliberate |

Agents **follow signal A** when hungry (`energy < ENERGY_HUNGRY_THRESHOLD × trait_signal_a_strength`), turning toward the strongest sensor gradient. This creates self-reinforcing food trails.

### Signal B — Danger / Depletion Marker
| Trigger | Strength | When |
|---------|----------|------|
| Agent dies | 0.5 | On death, at corpse position |
| Agent energy < 1.0 | 0.3 | Each step while low energy |
| Resource cell depleted (≤ 0.01 after consumption) | `SIGNAL_B_DEPLETION_EMIT` (0.05) | After consumption |
| Agent chooses ACTION_EMIT_B | 0.5 | Deliberate |

Agents **avoid signal B** when sensed above their per-agent `trait_signal_b_threshold`, turning away. They also **pay an energy cost** proportional to signal B strength at their position (`SIGNAL_B_METABOLIC_COST = 0.01`), giving real metabolic teeth to danger avoidance.

## Agent Design

### Sensors
Each agent samples three directions (forward, 45° left, 45° right):
- Resource level
- Signal A strength
- Signal B strength (x3 → 9 values) + local density + energy + age + memory → 16 inputs

### Actions (7 int-coded)
`MOVE`, `TURN_LEFT`, `TURN_RIGHT`, `EAT`, `EMIT_A`, `EMIT_B`, `REPRODUCE`

### Decision Tree (vectorized, no per-agent loop)
Priority order:
1. **Reproduce** (30% chance if `effective_energy ≥ ENERGY_REPRO_THRESHOLD`)
2. **Avoid danger** (turn away from signal B when above per-agent threshold)
3. **Forage** (when hungry and signal A trail detected: turn toward strongest gradient)
4. **Wander** (85% move, 15% random turn — default state)

### Heritable Traits (mutated on reproduction)
| Trait | Default | Effect |
|-------|---------|--------|
| speed | 1.0 | Multiply movement distance |
| sensing_radius | 1.0 | Multiply sensor distance |
| mutation_rate | 0.1 | Probability of each trait mutating |
| energy_efficiency | 1.0 | ×energy gain from food, ÷existence cost |
| signal_a_strength | 1.0 | ×trail emission strength, ×foraging hunger threshold |
| signal_b_threshold | 0.5 | Danger sensitivity (lower = more skittish) |

## Energy System

```
+ Gain:
  - consuming resources:   gain = consumed × ENERGY_GAIN_RESOURCE × trait_efficiency

- Loss:
  - movement:              cost = ENERGY_COST_MOVE × trait_speed
  - existence:             cost = ENERGY_COST_EXIST / trait_efficiency
  - signaling cost:        ENERGY_COST_SIGNAL if EMIT action
  - signal B zone cost:    cost = sigB_strength × SIGNAL_B_METABOLIC_COST
  - reproduction:          parent loses 50% energy

Rules:
  - Die if energy <= 0 or age > AGE_MAX (5000)
  - Reproduce if effective_energy >= ENERGY_REPRO_THRESHOLD (2.0)
  - Max energy: ENERGY_MAX = 3.0 (any surplus is clipped)
  - Resource consumption: each step, agents take 50% of cell value
  - Resources are finite per cell (consumable, not free)
```

## Performance Strategy

Use vectorized NumPy arrays (not Python objects per agent) for 10K–100K agent performance:

```python
positions = np.zeros((N, 2))
angles = np.zeros(N)
energy = np.ones(N)
memory = np.zeros((N, 4))
```

## Experiments

The `experiment.py` runner reads `experiments.yml` and runs headless simulations, saving results (CSV summaries, JSON history, PNG plots) under `results/run_NNN/`.

## Phases

- **Phase 1** ✅ — Minimal simulation with rule-based agents, signals, reproduction, death, visualization
- **Phase 2** ✅ — Heritable traits with mutation (speed, sensing, efficiency, etc.)
- **Phase 3** — Neural network policies (replace rule-based with tiny NN)
- **Phase 4** — Emergence experiments (clustering, trail formation, specialization)

## Sandbox Rules

Agents CAN:
- sense local world
- move
- emit signals
- consume resources
- reproduce

Agents CANNOT:
- access files
- access internet
- execute system calls
- inspect environment outside simulation
