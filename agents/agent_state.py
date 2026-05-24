import numpy as np
import config


TRAIT_NAMES = ['speed', 'sensing_radius', 'mutation_rate', 'energy_efficiency', 'signal_a_strength', 'signal_b_threshold']


class AgentState:
    def __init__(self, num_agents, max_capacity=None):
        self.max_capacity = max_capacity or (num_agents * 4)
        self.num_agents = num_agents

        self.positions = np.zeros((self.max_capacity, 2), dtype=np.float64)
        self.positions[:num_agents] = np.random.rand(num_agents, 2) * [config.WORLD_WIDTH, config.WORLD_HEIGHT]
        self.angles = np.zeros(self.max_capacity, dtype=np.float64)
        self.angles[:num_agents] = np.random.uniform(0, 2 * np.pi, size=num_agents)
        self.energy = np.zeros(self.max_capacity, dtype=np.float64)
        self.energy[:num_agents] = config.ENERGY_START
        self.age = np.zeros(self.max_capacity, dtype=np.int32)
        self.alive = np.zeros(self.max_capacity, dtype=bool)
        self.alive[:num_agents] = True
        self.memory = np.zeros((self.max_capacity, config.MEMORY_SIZE), dtype=np.float32)
        self.generation = np.zeros(self.max_capacity, dtype=np.int32)
        self.lineage = np.zeros(self.max_capacity, dtype=np.int32)

        self.trait_speed = np.ones(self.max_capacity, dtype=np.float64)
        self.trait_sensing_radius = np.ones(self.max_capacity, dtype=np.float64)
        self.trait_mutation_rate = np.full(self.max_capacity, config.MUTATION_RATE, dtype=np.float64)
        self.trait_energy_efficiency = np.ones(self.max_capacity, dtype=np.float64)
        self.trait_signal_a_strength = np.ones(self.max_capacity, dtype=np.float64)
        self.trait_signal_b_threshold = np.full(self.max_capacity, config.INITIAL_SIGNAL_B_THRESHOLD, dtype=np.float64)

        self._alive_cache = np.arange(num_agents)
        self._cache_valid = False

    def _invalidate_cache(self):
        self._cache_valid = False

    def compact(self):
        alive_mask = self.alive[:self.num_agents]
        if np.sum(~alive_mask) < self.num_agents * 0.3:
            return
        alive_indices = np.where(alive_mask)[0]
        n_alive = len(alive_indices)

        self.positions[:n_alive] = self.positions[alive_indices]
        self.angles[:n_alive] = self.angles[alive_indices]
        self.energy[:n_alive] = self.energy[alive_indices]
        self.age[:n_alive] = self.age[alive_indices]
        self.alive[:n_alive] = True
        self.alive[n_alive:self.num_agents] = False
        self.memory[:n_alive] = self.memory[alive_indices]
        self.generation[:n_alive] = self.generation[alive_indices]
        self.lineage[:n_alive] = self.lineage[alive_indices]
        self.trait_speed[:n_alive] = self.trait_speed[alive_indices]
        self.trait_sensing_radius[:n_alive] = self.trait_sensing_radius[alive_indices]
        self.trait_mutation_rate[:n_alive] = self.trait_mutation_rate[alive_indices]
        self.trait_energy_efficiency[:n_alive] = self.trait_energy_efficiency[alive_indices]
        self.trait_signal_a_strength[:n_alive] = self.trait_signal_a_strength[alive_indices]
        self.trait_signal_b_threshold[:n_alive] = self.trait_signal_b_threshold[alive_indices]

        self.num_agents = n_alive
        self._alive_cache = np.arange(n_alive)
        self._cache_valid = True

    def remove_dead(self):
        mask = self.alive[:self.num_agents] & (self.energy[:self.num_agents] <= 0)
        dead_positions = self.positions[:self.num_agents][mask].copy() if np.any(mask) else np.empty((0, 2))
        self.alive[:self.num_agents][mask] = False
        if np.any(mask):
            self._invalidate_cache()
        return dead_positions

    def get_alive_indices(self):
        if self._cache_valid:
            return self._alive_cache
        indices = np.where(self.alive[:self.num_agents])[0]
        self._alive_cache = indices
        self._cache_valid = True
        return indices

    def apply_actions(self, actions, alive_indices):
        if len(alive_indices) == 0:
            return

        speeds = self.trait_speed[alive_indices]
        efficiency = self.trait_energy_efficiency[alive_indices]

        move_mask = actions == config.ACTION_MOVE
        left_mask = actions == config.ACTION_TURN_LEFT
        right_mask = actions == config.ACTION_TURN_RIGHT

        self.angles[alive_indices[left_mask]] -= config.SENSOR_ANGLE
        self.angles[alive_indices[right_mask]] += config.SENSOR_ANGLE

        angles = self.angles[alive_indices]
        dx = np.cos(angles) * config.SPEED * speeds
        dy = np.sin(angles) * config.SPEED * speeds
        dx[~move_mask] = 0
        dy[~move_mask] = 0

        self.positions[alive_indices, 0] = (self.positions[alive_indices, 0] + dx) % config.WORLD_WIDTH
        self.positions[alive_indices, 1] = (self.positions[alive_indices, 1] + dy) % config.WORLD_HEIGHT

        energy_cost_move = config.ENERGY_COST_MOVE * speeds
        self.energy[alive_indices[move_mask]] -= energy_cost_move[move_mask]
        energy_cost_exist = config.ENERGY_COST_EXIST / efficiency
        self.energy[alive_indices] -= energy_cost_exist

    def consume_resources(self, consumed, efficiencies, indices):
        gain = consumed * config.ENERGY_GAIN_RESOURCE * efficiencies
        self.energy[indices] = np.clip(self.energy[indices] + gain, 0, config.ENERGY_MAX)

    def eat(self, eat_mask):
        self.energy[eat_mask] -= config.ENERGY_COST_EXIST

    def age_step(self):
        self.age[:self.num_agents] += 1
        old_mask = self.age[:self.num_agents] > config.AGE_MAX
        self.energy[:self.num_agents][old_mask] = 0

    def get_reproducers(self):
        return np.where(self.alive[:self.num_agents] & (self.energy[:self.num_agents] >= config.ENERGY_REPRO_THRESHOLD))[0]

    def reproduce(self, parent_indices):
        num_new = len(parent_indices)
        if num_new == 0:
            return

        available_slots = config.MAX_AGENTS - self.num_agents
        if available_slots <= 0:
            return

        actual_new = min(num_new, available_slots)
        parent_indices = parent_indices[:actual_new]

        if self.num_agents + actual_new > self.max_capacity:
            self.compact()
            if self.num_agents + actual_new > self.max_capacity:
                new_cap = self.max_capacity * 2
                self._resize(new_cap)

        start = self.num_agents
        end = self.num_agents + actual_new

        self.positions[start:end] = self.positions[parent_indices]
        self.angles[start:end] = self.angles[parent_indices]
        self.energy[start:end] = self.energy[parent_indices] * 0.5
        self.age[start:end] = 0
        self.alive[start:end] = True
        self.memory[start:end] = self.memory[parent_indices] + np.random.normal(0, 0.1, size=(actual_new, config.MEMORY_SIZE))
        self.generation[start:end] = self.generation[parent_indices] + 1
        self.lineage[start:end] = self.lineage[parent_indices] + 1

        trait_arrays = [
            self.trait_speed,
            self.trait_sensing_radius,
            self.trait_mutation_rate,
            self.trait_energy_efficiency,
            self.trait_signal_a_strength,
            self.trait_signal_b_threshold,
        ]

        for arr in trait_arrays:
            arr[start:end] = arr[parent_indices]

        self._mutate_traits(start, end)

        self.num_agents += actual_new
        self.energy[parent_indices] *= 0.5
        self._invalidate_cache()

    def _mutate_traits(self, start, end):
        n = end - start

        mutation_rates = self.trait_mutation_rate[start:end]
        should_mutate = np.random.random((n, len(TRAIT_NAMES))) < mutation_rates[:, None]

        noise = np.random.normal(0, config.MUTATION_STRENGTH, (n, len(TRAIT_NAMES)))
        trait_arrays = [
            self.trait_speed,
            self.trait_sensing_radius,
            self.trait_mutation_rate,
            self.trait_energy_efficiency,
            self.trait_signal_a_strength,
            self.trait_signal_b_threshold,
        ]

        for i, arr in enumerate(trait_arrays):
            mask = should_mutate[:, i]
            arr[start:end][mask] += noise[mask, i]
            arr[start:end] = np.clip(arr[start:end], 0.1, 3.0)

    def _resize(self, new_capacity):
        self.positions = np.vstack([self.positions, np.zeros((new_capacity - self.max_capacity, 2))])
        self.angles = np.hstack([self.angles, np.zeros(new_capacity - self.max_capacity)])
        self.energy = np.hstack([self.energy, np.zeros(new_capacity - self.max_capacity)])
        self.age = np.hstack([self.age, np.zeros(new_capacity - self.max_capacity, dtype=np.int32)])
        self.alive = np.hstack([self.alive, np.zeros(new_capacity - self.max_capacity, dtype=bool)])
        self.memory = np.vstack([self.memory, np.zeros((new_capacity - self.max_capacity, config.MEMORY_SIZE))])
        self.generation = np.hstack([self.generation, np.zeros(new_capacity - self.max_capacity, dtype=np.int32)])
        self.lineage = np.hstack([self.lineage, np.zeros(new_capacity - self.max_capacity, dtype=np.int32)])
        self.trait_speed = np.hstack([self.trait_speed, np.ones(new_capacity - self.max_capacity)])
        self.trait_sensing_radius = np.hstack([self.trait_sensing_radius, np.ones(new_capacity - self.max_capacity)])
        self.trait_mutation_rate = np.hstack([self.trait_mutation_rate, np.full(new_capacity - self.max_capacity, config.MUTATION_RATE)])
        self.trait_energy_efficiency = np.hstack([self.trait_energy_efficiency, np.ones(new_capacity - self.max_capacity)])
        self.trait_signal_a_strength = np.hstack([self.trait_signal_a_strength, np.ones(new_capacity - self.max_capacity)])
        self.trait_signal_b_threshold = np.hstack([self.trait_signal_b_threshold, np.full(new_capacity - self.max_capacity, config.INITIAL_SIGNAL_B_THRESHOLD)])
        self.max_capacity = new_capacity
