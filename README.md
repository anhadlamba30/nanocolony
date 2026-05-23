# Nanocolony Lab — Artificial Life Simulation

An artificial life sandbox where thousands of autonomous agents compete for resources, emit signals, and create emergent behaviors like trail-following and panic dynamics.

## Visuals

[![Simulation preview](assets/poster.png)](assets/simulation.mp4)
*Three experiments running side-by-side: abundance (explosive growth), sustained (steady growth), and fast reproduction (boom/bust). Click the image to view the animation.*

![Experiment results](assets/experiment_plots.png)
*Population, energy, signals, and births/deaths across the three environments*

## Quick Start

```bash
conda env create -f environment.yml
conda activate nanocolony
python main.py
```

**ESC** to quit. Watch the simulation evolve!

## What You're Seeing

| Color | Meaning |
|-------|---------|
| **Green** | Resources (food) - agents consume these to gain energy |
| **Blue** | Signal A - "food trail" pheromone left by agents |
| **Red** | Signal B - "danger" signal from low-energy agents |
| **Bright dots** | High-energy agents (healthy) |
| **Dim dots** | Low-energy agents (struggling) |

## Key Metrics

- **Population**: Number of living agents
- **Births/Deaths**: Per step and cumulative
- **Avg energy**: Health of the population (0 = dead, 2.0 = max)
- **Avg resource**: How much food is available globally
- **Avg signal A/B**: Intensity of communication signals

## Parameters to Tweak (`config.py`)

### Resource Balance
| Parameter | Effect |
|-----------|--------|
| `RESOURCE_SPAWN_RATE` | Higher = more food spawns |
| `RESOURCE_MAX` | Maximum resource value per cell |
| `ENERGY_GAIN_RESOURCE` | How much energy per food consumed |
| `ENERGY_COST_EXIST` | Energy lost per step (higher = harder survival) |

### Reproduction
| Parameter | Effect |
|-----------|--------|
| `ENERGY_REPRO_THRESHOLD` | Energy needed to reproduce (lower = faster growth) |
| `ENERGY_START` | Starting energy for new agents |

### Signals
| Parameter | Effect |
|-----------|--------|
| `SIGNAL_DECAY` | How fast signals fade (lower = shorter trails) |
| `SIGNAL_DIFFUSE_RATE` | How far signals spread |

### World
| Parameter | Effect |
|-----------|--------|
| `INITIAL_AGENTS` | Starting population |
| `WORLD_WIDTH`, `WORLD_HEIGHT` | Map size |

## Example Configurations

### Balanced Ecosystem (Default)
```python
RESOURCE_SPAWN_RATE = 0.004
ENERGY_REPRO_THRESHOLD = 1.5
ENERGY_COST_EXIST = 0.002
```
Population stabilizes around 100-500, sustainable.

### Explosion (High Resources)
```python
RESOURCE_SPAWN_RATE = 0.01
ENERGY_REPRO_THRESHOLD = 1.3
ENERGY_COST_EXIST = 0.001
```
Population explodes to millions, screen turns blue/purple.

### Collapse (Scarcity)
```python
RESOURCE_SPAWN_RATE = 0.001
ENERGY_REPRO_THRESHOLD = 1.8
ENERGY_COST_EXIST = 0.005
```
Population dies out quickly.

## Conducting Experiments

### Headless Experiment Runner

```bash
python experiment.py --list           # List available experiments
python experiment.py                  # Run all experiments
python experiment.py --experiments food_scarcity  # Run specific experiment
python experiment.py --steps 1000     # Override max steps
```

### 1. Signal Response Experiment
Currently, agents:
- **Follow signal A** (move toward blue trails)
- **Avoid signal B** (turn away from red zones)
- **Emit signal B** when low energy (distress calls)

Try modifying `agent_logic.py` to change behavior:
- Make agents ignore signals
- Add more signal types (C = territory, D = food source type)
- Add signal response based on energy level

### 2. Evolution Experiment
Track lineage by modifying `agent_state.py`:
- Add `parent_id` to each agent
- Record which genomes survive longer
- Observe if specific behaviors dominate

### 3. Predator Experiment
Add a "predator" agent type:
- Emits strong signal B
- Hunts nearest agent
- Dies if it doesn't eat periodically

### 4. Resource Hotspots
Modify `environment.py` to create:
- Moving food sources (rivers)
- Depleting patches (ore veins)
- Seasonal changes

## Interpreting Results

### Signs of Healthy Ecosystem
- Population fluctuates but stays bounded
- Both births and deaths happening
- Avg energy between 0.4-0.8
- No signal dominates completely

### Signs of Problems
- **Population = 0**: Too harsh, no survivors
- **Population growing forever**: Resources too generous
- **Signal A = 0, Signal B high**: Panic mode, system stressed
- **Zero deaths**: Not realistic, balance is off

### Interesting Phenomena to Look For
1. **Trail formation**: Agents following each other to food
2. **Clustering**: Agents grouping together
3. **Oscillations**: Population boom/bust cycles
4. **Signal arms race**: Different signals competing

## Project Structure

```
nanocolony/
├── main.py              # Entry point
├── config.py            # All tunable parameters
├── experiment.py        # Headless experiment runner
├── world/
│   └── environment.py   # 512x512 grid, resources, signals
├── agents/
│   ├── agent_state.py   # Vectorized agent data
│   └── agent_logic.py   # Sensing and decision logic
├── simulation/
│   └── step.py          # Main simulation loop
└── rendering/
    └── renderer.py      # Pygame visualization
```

## Performance Tips

- Population >50K slows down (Python limitation)
- Rendering samples max 5000 agents for display
- See `ARCHITECTURE.md` for detailed design notes

## Future Enhancements

- Phase 2: Evolution with heritable traits
- Phase 3: Neural network policies
- GPU acceleration with PyTorch/JAX
- 3D extension

## License

MIT
