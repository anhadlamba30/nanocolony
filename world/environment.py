import numpy as np
import config


class Environment:
    def __init__(self):
        self.width = config.WORLD_WIDTH
        self.height = config.WORLD_HEIGHT
        self.resources = np.zeros((self.height, self.width), dtype=np.float32)
        self.signal_a = np.zeros((self.height, self.width), dtype=np.float32)
        self.signal_b = np.zeros((self.height, self.width), dtype=np.float32)
        self._spawn_initial_resources()

    def _spawn_initial_resources(self):
        x = np.random.randint(0, self.width, size=1000)
        y = np.random.randint(0, self.height, size=1000)
        self.resources[y, x] = np.random.uniform(0.3, 0.5, size=1000)

    def update(self):
        self.resources *= config.RESOURCE_DECAY
        self.signal_a *= config.SIGNAL_DECAY
        self.signal_b *= config.SIGNAL_DECAY
        self._diffuse_signals()
        self._spawn_resources()

    def _diffuse_signals(self):
        r = config.SIGNAL_DIFFUSE_RATE
        a = self.signal_a
        self.signal_a = np.clip(
            a * (1 - 4*r) +
            r * (np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0) + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)),
            0, 1.0
        )
        b = self.signal_b
        self.signal_b = np.clip(
            b * (1 - 4*r) +
            r * (np.roll(b, 1, axis=0) + np.roll(b, -1, axis=0) + np.roll(b, 1, axis=1) + np.roll(b, -1, axis=1)),
            0, 1.0
        )

    def _spawn_resources(self):
        n_spawn = int(self.height * self.width * config.RESOURCE_SPAWN_RATE)
        if n_spawn == 0:
            return
        y = np.random.randint(0, self.height, size=n_spawn)
        x = np.random.randint(0, self.width, size=n_spawn)
        self.resources[y, x] = np.random.uniform(0.1, config.RESOURCE_MAX, size=n_spawn)

    def emit_signals_batch(self, y_arr, x_arr, signal_type, strengths):
        if len(y_arr) == 0:
            return
        y_arr = np.clip(y_arr, 0, self.height - 1)
        x_arr = np.clip(x_arr, 0, self.width - 1)
        if signal_type == 'a':
            np.add.at(self.signal_a, (y_arr, x_arr), strengths)
            np.clip(self.signal_a, 0, 1.0, out=self.signal_a)
        else:
            np.add.at(self.signal_b, (y_arr, x_arr), strengths)
            np.clip(self.signal_b, 0, 1.0, out=self.signal_b)

    def emit_signal(self, y, x, signal_type, strength=1.0):
        if signal_type == 'a':
            self.signal_a[y, x] = np.clip(self.signal_a[y, x] + strength, 0, 1.0)
        else:
            self.signal_b[y, x] = np.clip(self.signal_b[y, x] + strength, 0, 1.0)

    def sample_resources(self, positions):
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        return self.resources[y, x]

    def sample_signals(self, positions):
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        return self.signal_a[y, x], self.signal_b[y, x]
