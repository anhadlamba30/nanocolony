import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import subprocess
import config
from simulation.step import Simulation

plt.style.use('dark_background')

EXPERIMENTS = [
    ('Abundance', {
        'RESOURCE_SPAWN_RATE': 0.008,
        'ENERGY_REPRO_THRESHOLD': 1.3,
        'ENERGY_COST_EXIST': 0.001,
    }),
    ('Sustained', {
        'RESOURCE_SPAWN_RATE': 0.006,
        'ENERGY_REPRO_THRESHOLD': 1.4,
        'ENERGY_COST_EXIST': 0.002,
    }),
    ('Fast Repro', {
        'RESOURCE_SPAWN_RATE': 0.004,
        'ENERGY_REPRO_THRESHOLD': 1.2,
        'ENERGY_COST_EXIST': 0.002,
    }),
]

STEPS = 1000
INTERVAL = 20


def apply_params(params):
    for k, v in params.items():
        setattr(config, k, v)


def save_config():
    return {k: getattr(config, k) for k in
            ['RESOURCE_SPAWN_RATE', 'ENERGY_REPRO_THRESHOLD', 'ENERGY_COST_EXIST',
             'ENERGY_COST_MOVE', 'ENERGY_COST_SIGNAL']}


def render_composite(sims, names, step):
    n = len(sims)
    fig = plt.figure(figsize=(12, 4), facecolor='#0a0a0a')

    panel_w = 0.283
    panel_h = 0.85
    left0 = 0.055
    wspace = 0.02
    bottom = 0.02

    for i in range(n):
        left = left0 + i * (panel_w + wspace)
        ax = fig.add_axes([left, bottom, panel_w, panel_h])
        ax.set_facecolor('#0a0a0a')
        state = sims[i].get_state_for_render()

        r = np.clip(state['signal_b'] * 2.0, 0, 1)
        g = np.clip(state['resources'] / config.RESOURCE_MAX * 1.5, 0, 1)
        b = np.clip(state['signal_a'] * 2.0, 0, 1)
        bg = np.stack([r, g, b], axis=-1)
        bg = bg ** 0.65

        ax.imshow(bg, interpolation='bilinear', aspect='auto')
        ax.set_xlim(0, config.WORLD_WIDTH)
        ax.set_ylim(config.WORLD_HEIGHT, 0)
        ax.autoscale(False)

        alive = state['alive']
        pos = state['positions'][alive]
        energy = state['energy'][alive]

        if len(pos) > 0:
            e_norm = energy / config.ENERGY_MAX
            sizes = (1 + e_norm * 5) ** 2
            colors = plt.cm.magma(e_norm)

            ax.scatter(pos[:, 0], pos[:, 1],
                       c='white', s=sizes * 2, alpha=0.06,
                       edgecolors='none', zorder=3)

            ax.scatter(pos[:, 0], pos[:, 1],
                       c=colors, s=sizes, alpha=0.85,
                       edgecolors='white', linewidths=0.1, zorder=5)

        pop = len(pos)
        avg_e = np.mean(energy) if pop > 0 else 0
        ax.set_title(f'{names[i]}  •  {pop:>5}  •  ø{avg_e:.2f}',
                     fontsize=9, color='#cccccc', pad=2, family='monospace')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#2a2a2a')
            spine.set_linewidth(0.5)

    fig.text(0.5, 0.96, f'step {step}', fontsize=11, color='#666666',
             family='monospace', ha='center', va='top')
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor='#0a0a0a', edgecolor='none')
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_img


if __name__ == '__main__':
    saved = save_config()

    sims = []
    for name, params in EXPERIMENTS:
        apply_params(params)
        sims.append(Simulation())
    apply_params(saved)

    names = [e[0] for e in EXPERIMENTS]

    frames = []
    for step in range(STEPS):
        for i, (_, params) in enumerate(EXPERIMENTS):
            apply_params(params)
            sims[i].step()
        apply_params(saved)

        if step % INTERVAL == 0:
            frames.append(render_composite(sims, names, step))
            if step % 200 == 0:
                print(f'Step {step}/{STEPS} ({len(frames)} frames)', flush=True)

    tmpdir = 'assets/.frames'
    os.makedirs(tmpdir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f'{tmpdir}/f_{i:04d}.png')
    mp4_path = 'assets/simulation.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '8',
        '-i', f'{tmpdir}/f_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '28', '-movflags', '+faststart',
        mp4_path
    ], capture_output=True)
    for f_name in os.listdir(tmpdir):
        os.remove(f'{tmpdir}/{f_name}')
    os.rmdir(tmpdir)
    size_kb = os.path.getsize(mp4_path) // 1024
    print(f'MP4: {mp4_path} ({size_kb} KB)', flush=True)
