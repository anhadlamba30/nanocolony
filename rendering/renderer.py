import pygame
import numpy as np
import config


class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scale = config.RENDER_SCALE

        pygame.init()
        self.screen = pygame.display.set_mode(
            (width * self.scale, height * self.scale)
        )
        pygame.display.set_caption("Nanocolony Lab")
        self.clock = pygame.time.Clock()

        self.display = pygame.Surface((width, height))

    def render(self, state):
        self.display.fill((0, 0, 0))

        resources = state['resources']
        sig_a = state['signal_a']
        sig_b = state['signal_b']

        r_channel = (np.clip(sig_b * 255, 0, 255)).astype(np.uint8)
        g_channel = (np.clip(resources * 255, 0, 255)).astype(np.uint8)
        b_channel = (np.clip(sig_a * 255, 0, 255)).astype(np.uint8)

        rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        rgb = np.transpose(rgb, (1, 0, 2))

        pygame.surfarray.blit_array(self.display, rgb)

        positions = state['positions'][state['alive']]
        energy = state['energy'][state['alive']]

        if len(positions) > 0:
            max_render = min(len(positions), 5000)
            if len(positions) > max_render:
                idx = np.random.choice(len(positions), max_render, replace=False)
                positions = positions[idx]
                energy = energy[idx]

            positions_int = positions.astype(np.int32)
            for i, (x, y) in enumerate(positions_int):
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self._energy_color(energy[i])
                    self.display.set_at((x, y), color)

        scaled = pygame.transform.scale(self.display, (self.width * self.scale, self.height * self.scale))
        self.screen.blit(scaled, (0, 0))

        font = pygame.font.Font(None, 24)
        text = font.render(f"Pop: {state['population']} Step: {state['step']}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(config.TARGET_FPS)

    def _energy_color(self, energy):
        e = np.clip(energy / config.ENERGY_MAX, 0, 1)
        r = int(255 * e)
        g = int(100 * (1 - e))
        b = int(50 + 100 * e)
        return (r, g, b)

    def close(self):
        pygame.quit()

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False