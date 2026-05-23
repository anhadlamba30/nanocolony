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
    # 1. Update signal diffusion
    diffuse_fields()

    # 2. For each agent:
    for agent in agents:
        inputs = sense_environment(agent)
        actions = policy(inputs)
        apply_actions(agent, actions)

    # 3. Resource update
    spawn_resources()

    # 4. Lifecycle updates
    handle_reproduction()
    handle_death()

    # 5. Logging
    record_metrics()

    # 6. Render frame
    draw()
```

## Signal Fields (Key for Emergence)

Each field diffuses across the grid, decays over time, and is written by agents:

```python
field[x, y] += emission_strength
field *= decay_rate
field = diffuse(field)
```

Types:
- **Signal A** — "food trail" pheromone
- **Signal B** — "danger" distress signal

## Agent Design

### Sensors
Each agent samples:
- Resource levels (forward, left, right)
- Signal A & B (forward, left, right)
- Local agent density
- Current energy level

### Actions
- move forward
- rotate left/right
- eat resource
- emit signal A
- emit signal B
- reproduce (if energy threshold met)

### Heritable Traits
Traits mutated on reproduction:
- speed
- sensing radius
- mutation rate
- energy efficiency
- signal A strength
- signal B threshold

## Energy System

```
+ Gain:
  - consuming resources

- Loss:
  - movement
  - existence per step
  - signaling cost

Rules:
  - Die if energy <= 0
  - Reproduce if energy >= threshold
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
