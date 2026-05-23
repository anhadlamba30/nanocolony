import numpy as np
import config
from agents.agent_state import AgentState
from agents.agent_logic import AgentLogic
from world.environment import Environment


class Simulation:
    def __init__(self):
        self.env = Environment()
        self.agent_state = AgentState(config.INITIAL_AGENTS)
        self.agent_logic = AgentLogic(self.env)
        self.step_count = 0
        self.stats = {
            'population': [],
            'avg_energy': [],
            'births': 0,
            'deaths': 0,
            'step_births': 0,
            'step_deaths': 0,
            'avg_signal_a': [],
            'avg_signal_b': [],
            'avg_resource': [],
            'avg_traits': []
        }
        self.prev_pop = config.INITIAL_AGENTS

    def step(self):
        self.stats['step_births'] = 0
        self.stats['step_deaths'] = 0

        self.env.update()

        inputs, alive_indices = self.agent_logic.sense(self.agent_state)

        actions = self.agent_logic.compute_actions(inputs, self.agent_state, alive_indices)

        self.agent_state.apply_actions(actions, alive_indices)

        reproducers = self.agent_logic.process_actions(self.agent_state, actions, alive_indices)

        if len(reproducers) > 0:
            old_count = self.agent_state.num_agents
            self.agent_state.reproduce(reproducers)
            new_births = self.agent_state.num_agents - old_count
            self.stats['births'] += new_births
            self.stats['step_births'] = new_births

        prev_alive = int(np.sum(self.agent_state.alive[:self.agent_state.num_agents]))
        self.agent_state.age_step()

        self.agent_state.remove_dead()

        if self.step_count % 200 == 0:
            self.agent_state.compact()

        new_deaths = prev_alive - int(np.sum(self.agent_state.alive[:self.agent_state.num_agents]))
        self.stats['step_deaths'] = new_deaths
        self.stats['deaths'] += new_deaths

        alive_mask = self.agent_state.alive[:self.agent_state.num_agents]
        alive_count = int(np.sum(alive_mask))
        self.stats['population'].append(alive_count)
        if alive_count > 0:
            alive = self.agent_state.energy[:self.agent_state.num_agents][alive_mask]
            self.stats['avg_energy'].append(np.mean(alive))
        else:
            self.stats['avg_energy'].append(0.0)

        self.stats['avg_signal_a'].append(np.mean(self.env.signal_a))
        self.stats['avg_signal_b'].append(np.mean(self.env.signal_b))
        self.stats['avg_resource'].append(np.mean(self.env.resources))

        if self.step_count % 100 == 0:
            if alive_count > 0:
                avg_traits = {
                    'speed': np.mean(self.agent_state.trait_speed[:self.agent_state.num_agents][alive_mask]),
                    'sensing_radius': np.mean(self.agent_state.trait_sensing_radius[:self.agent_state.num_agents][alive_mask]),
                    'energy_efficiency': np.mean(self.agent_state.trait_energy_efficiency[:self.agent_state.num_agents][alive_mask]),
                }
                self.stats['avg_traits'].append(avg_traits)
            else:
                self.stats['avg_traits'].append({})

        self.step_count += 1

    def get_state_for_render(self):
        n = self.agent_state.num_agents
        return {
            'positions': self.agent_state.positions[:n],
            'energy': self.agent_state.energy[:n],
            'alive': self.agent_state.alive[:n],
            'resources': self.env.resources,
            'signal_a': self.env.signal_a,
            'signal_b': self.env.signal_b,
            'step': self.step_count,
            'population': int(np.sum(self.agent_state.alive[:n]))
        }
