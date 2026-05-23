import config
from simulation.step import Simulation
from rendering.renderer import Renderer


def main():
    sim = Simulation()
    renderer = Renderer(config.WORLD_WIDTH, config.WORLD_HEIGHT)

    print(f"=" * 60)
    print(f"NANOCOLONY LAB - Artificial Life Simulation")
    print(f"=" * 60)
    print(f"Initial agents: {config.INITIAL_AGENTS}")
    print(f"World size: {config.WORLD_WIDTH}x{config.WORLD_HEIGHT}")
    print(f"Resource spawn: {config.RESOURCE_SPAWN_RATE}")
    print(f"Repro threshold: {config.ENERGY_REPRO_THRESHOLD}")
    print(f"=" * 60)

    running = True
    while running:
        sim.step()

        state = sim.get_state_for_render()
        renderer.render(state)

        if sim.step_count % 50 == 0:
            stats = sim.stats
            print(f"\n--- Step {sim.step_count} ---")
            print(f"Population: {state['population']:,}")
            print(f"  Births (step): {stats['step_births']:,}")
            print(f"  Deaths (step): {stats['step_deaths']:,}")
            print(f"  Net change: {stats['step_births'] - stats['step_deaths']:+d}")
            print(f"  Total births: {stats['births']:,}")
            print(f"  Total deaths: {stats['deaths']:,}")
            if stats['avg_energy']:
                print(f"Avg energy: {stats['avg_energy'][-1]:.3f}")
            if stats['avg_resource']:
                print(f"Avg resource: {stats['avg_resource'][-1]:.4f}")
            if stats['avg_signal_a']:
                print(f"Avg signal A: {stats['avg_signal_a'][-1]:.4f}")
            if stats['avg_signal_b']:
                print(f"Avg signal B: {stats['avg_signal_b'][-1]:.4f}")

        if state['population'] == 0:
            print("\nAll agents died!")
            break

        running = not renderer.check_quit()

    renderer.close()
    print(f"\n{'=' * 60}")
    print(f"SIMULATION ENDED")
    print(f"{'=' * 60}")
    print(f"Total steps: {sim.step_count}")
    print(f"Final population: {state['population']:,}")
    print(f"Total births: {sim.stats['births']:,}")
    print(f"Total deaths: {sim.stats['deaths']:,}")


if __name__ == "__main__":
    main()