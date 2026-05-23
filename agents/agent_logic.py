import numpy as np
import config


class AgentLogic:
    def __init__(self, env):
        self.env = env

    def sense(self, agent_state):
        alive_indices = agent_state.get_alive_indices()
        if len(alive_indices) == 0:
            return np.zeros((0, 16)), []

        positions = agent_state.positions[alive_indices]
        angles = agent_state.angles[alive_indices]
        sensing_radii = agent_state.trait_sensing_radius[alive_indices]

        sensors = self._sample_sensors(positions, angles, sensing_radii)

        energy = agent_state.energy[alive_indices].reshape(-1, 1)
        age = agent_state.age[alive_indices].reshape(-1, 1) / config.AGE_MAX
        memory = agent_state.memory[alive_indices]

        inputs = np.hstack([sensors, energy, age, memory])
        return inputs, alive_indices

    def _sample_sensors(self, positions, angles, sensing_radii=None):
        n = len(positions)
        if n == 0:
            return np.zeros((n, 10))

        if sensing_radii is None:
            sensing_radii = np.ones(n)

        angle_offsets = np.array([0, -config.SENSOR_ANGLE, config.SENSOR_ANGLE])

        base_distance = config.SENSOR_DISTANCE
        sensor_distances = base_distance * sensing_radii

        sensor_angles = angles[:, None] + angle_offsets[None, :]
        sample_x = (positions[:, 0:1] + sensor_distances[:, None] * np.cos(sensor_angles)) % config.WORLD_WIDTH
        sample_y = (positions[:, 1:2] + sensor_distances[:, None] * np.sin(sensor_angles)) % config.WORLD_HEIGHT

        sample_y = np.clip(sample_y.astype(int), 0, config.WORLD_HEIGHT - 1)
        sample_x = np.clip(sample_x.astype(int), 0, config.WORLD_WIDTH - 1)

        res = self.env.resources[sample_y, sample_x]
        sig_a = self.env.signal_a[sample_y, sample_x]
        sig_b = self.env.signal_b[sample_y, sample_x]

        sensor_data = np.stack([res, sig_a, sig_b], axis=2).reshape(n, 9)
        density = self._compute_density(positions)
        return np.hstack([sensor_data, density.reshape(-1, 1)])

    def _compute_density(self, positions):
        n = len(positions)
        if n == 0:
            return np.array([])
        cell_size = 32
        grid_w = config.WORLD_WIDTH // cell_size
        grid_h = config.WORLD_HEIGHT // cell_size
        cell_x = np.clip((positions[:, 0] // cell_size).astype(int), 0, grid_w - 1)
        cell_y = np.clip((positions[:, 1] // cell_size).astype(int), 0, grid_h - 1)
        cell_indices = cell_y * grid_w + cell_x
        counts = np.bincount(cell_indices, minlength=grid_w * grid_h)
        return counts[cell_indices] / max(n, 1)

    def compute_actions(self, inputs, agent_state, alive_indices):
        n = len(inputs)
        if n == 0:
            return np.zeros(0, dtype=np.int32)

        speeds = agent_state.trait_speed[alive_indices]
        sensing = agent_state.trait_sensing_radius[alive_indices]
        efficiency = agent_state.trait_energy_efficiency[alive_indices]
        signal_b_threshold = agent_state.trait_signal_b_threshold[alive_indices]

        energy = inputs[:, 10]

        res_f, res_l, res_r = inputs[:, 0], inputs[:, 3], inputs[:, 6]
        sig_a_f, sig_a_l, sig_a_r = inputs[:, 1], inputs[:, 4], inputs[:, 7]
        sig_b_f, sig_b_l, sig_b_r = inputs[:, 2], inputs[:, 5], inputs[:, 8]

        max_res = np.maximum(np.maximum(res_f, res_l), res_r)
        has_resource = max_res > 0.05

        max_sig_a = np.maximum(np.maximum(sig_a_f, sig_a_l), sig_a_r)
        max_sig_b = np.maximum(np.maximum(sig_b_f, sig_b_l), sig_b_r)

        following_a = max_sig_a > 0.05
        avoiding_b = max_sig_b > signal_b_threshold

        effective_energy = energy * efficiency
        can_repro = effective_energy >= config.ENERGY_REPRO_THRESHOLD

        rand = np.random.random(n)
        rand2 = np.random.random(n)

        actions = np.select([
            can_repro & (rand < 0.3),
            has_resource & (rand < 0.7),
            has_resource & (rand >= 0.7) & (rand < 0.85),
            has_resource & (rand >= 0.85) & (rand2 < 0.5),
            has_resource & (rand >= 0.85),
            avoiding_b & (rand < 0.4),
            avoiding_b & (rand >= 0.4) & (rand2 < 0.5),
            avoiding_b & (rand >= 0.4),
            following_a & ~has_resource & (rand < 0.7),
            following_a & ~has_resource & (rand >= 0.7) & (rand < 0.8),
            ~has_resource & ~following_a & ~avoiding_b & (rand < 0.5),
            ~has_resource & ~following_a & ~avoiding_b & (rand >= 0.5) & (rand < 0.6),
            ~has_resource & ~following_a & ~avoiding_b & (rand >= 0.6) & (rand < 0.7),
            ~has_resource & ~following_a & ~avoiding_b & (rand >= 0.7) & (rand < 0.8),
            ~has_resource & ~following_a & ~avoiding_b & (rand >= 0.8) & (rand2 < 0.5),
            ~has_resource & ~following_a & ~avoiding_b & (rand >= 0.8),
        ], [
            config.ACTION_REPRODUCE,
            config.ACTION_MOVE,
            config.ACTION_EAT,
            config.ACTION_TURN_LEFT,
            config.ACTION_TURN_RIGHT,
            config.ACTION_TURN_LEFT,
            config.ACTION_TURN_RIGHT,
            config.ACTION_EMIT_B,
            config.ACTION_MOVE,
            config.ACTION_EMIT_A,
            config.ACTION_MOVE,
            config.ACTION_EMIT_A,
            config.ACTION_EMIT_B,
            config.ACTION_TURN_LEFT,
            config.ACTION_TURN_RIGHT,
            config.ACTION_EMIT_A,
        ], default=config.ACTION_MOVE)

        return actions

    def process_actions(self, agent_state, actions, alive_indices):
        if len(alive_indices) == 0:
            return np.array([])

        positions = agent_state.positions[alive_indices]

        resources = self.env.sample_resources(positions)
        consumed = resources * 0.5
        agent_state.consume_resources(consumed, alive_indices)

        signal_a_strengths = agent_state.trait_signal_a_strength[alive_indices]

        low_energy_mask = agent_state.energy[alive_indices] < 1.0
        if np.any(low_energy_mask):
            low_positions = positions[low_energy_mask]
            low_y = np.clip(low_positions[:, 1].astype(int), 0, config.WORLD_HEIGHT - 1)
            low_x = np.clip(low_positions[:, 0].astype(int), 0, config.WORLD_WIDTH - 1)
            self.env.emit_signals_batch(low_y, low_x, 'b', np.full(np.sum(low_energy_mask), 0.3))

        emit_a_mask = actions == config.ACTION_EMIT_A
        emit_b_mask = actions == config.ACTION_EMIT_B

        if np.any(emit_a_mask):
            emit_a_positions = positions[emit_a_mask]
            emit_a_y = np.clip(emit_a_positions[:, 1].astype(int), 0, config.WORLD_HEIGHT - 1)
            emit_a_x = np.clip(emit_a_positions[:, 0].astype(int), 0, config.WORLD_WIDTH - 1)
            strengths = signal_a_strengths[emit_a_mask] * 0.5
            self.env.emit_signals_batch(emit_a_y, emit_a_x, 'a', strengths)

        if np.any(emit_b_mask):
            emit_b_positions = positions[emit_b_mask]
            emit_b_y = np.clip(emit_b_positions[:, 1].astype(int), 0, config.WORLD_HEIGHT - 1)
            emit_b_x = np.clip(emit_b_positions[:, 0].astype(int), 0, config.WORLD_WIDTH - 1)
            self.env.emit_signals_batch(emit_b_y, emit_b_x, 'b', np.full(np.sum(emit_b_mask), 0.5))

        reproducers = alive_indices[actions == config.ACTION_REPRODUCE]
        return reproducers
