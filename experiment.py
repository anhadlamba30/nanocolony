import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
import config
from tqdm import tqdm
from simulation.step import Simulation

DEFAULT_EXPERIMENTS_FILE = 'experiments.yml'


def load_experiments(filepath=None):
    if filepath is None:
        filepath = DEFAULT_EXPERIMENTS_FILE

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, using empty experiment list.")
        return {}, {'max_steps': 500, 'early_stop_pop': 5}

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    defaults = data.get('defaults', {})
    experiments = {}

    for name, exp_data in data.get('experiments', {}).items():
        experiments[name] = {
            'description': exp_data.get('description', ''),
            'params': exp_data.get('params', {}),
        }

    return experiments, defaults


class Experiment:
    def __init__(self, name, params_override=None):
        self.name = name
        self.params_override = params_override or {}
        self.results = None

    def run(self, max_steps=500, early_stop_pop=10):
        original_config = {k: getattr(config, k) for k in dir(config) if k.isupper()}

        for key, value in self.params_override.items():
            if hasattr(config, key):
                setattr(config, key, value)

        sim = Simulation()
        history = {
            'step': [],
            'population': [],
            'births': [],
            'deaths': [],
            'avg_energy': [],
            'avg_resource': [],
            'avg_signal_a': [],
            'avg_signal_b': [],
            'avg_traits': []
        }

        for step in tqdm(range(max_steps), desc=f"  {self.name}", leave=False, unit='step'):
            sim.step()

            stats = sim.stats
            history['step'].append(sim.step_count)
            history['population'].append(stats['population'][-1] if stats['population'] else 0)
            history['births'].append(stats.get('step_births', 0))
            history['deaths'].append(stats.get('step_deaths', 0))
            history['avg_energy'].append(stats['avg_energy'][-1] if stats['avg_energy'] else 0)
            history['avg_resource'].append(stats['avg_resource'][-1] if stats['avg_resource'] else 0)
            history['avg_signal_a'].append(stats['avg_signal_a'][-1] if stats['avg_signal_a'] else 0)
            history['avg_signal_b'].append(stats['avg_signal_b'][-1] if stats['avg_signal_b'] else 0)
            history['avg_traits'].append(stats['avg_traits'][-1] if stats['avg_traits'] else {})

            if stats['population'][-1] <= early_stop_pop:
                tqdm.write(f"  [{self.name}] Early stop at step {sim.step_count} (pop={stats['population'][-1]})")
                break

        for key, value in original_config.items():
            setattr(config, key, value)

        self.results = history
        return history

    def summary(self):
        if not self.results:
            return {}
        pop = np.array(self.results['population'])
        birth = np.array(self.results['births'])
        death = np.array(self.results['deaths'])

        final_traits = self.results['avg_traits'][-1] if self.results['avg_traits'] else {}
        final_traits = {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in final_traits.items()}

        return {
            'name': self.name,
            'final_pop': int(pop[-1]) if len(pop) > 0 else 0,
            'peak_pop': int(pop.max()) if len(pop) > 0 else 0,
            'total_births': int(np.sum(birth)),
            'total_deaths': int(np.sum(death)),
            'steps_run': len(pop),
            'avg_pop': float(np.mean(pop)) if len(pop) > 0 else 0.0,
            'final_energy': float(self.results['avg_energy'][-1]) if self.results['avg_energy'] else 0.0,
            'final_traits': final_traits,
        }


def run_experiments(experiment_names=None, max_steps=500, early_stop_pop=5, experiments_file=None):
    experiments, defaults = load_experiments(experiments_file)

    if not experiments:
        print("No experiments defined.")
        return [], {}

    if experiment_names is None:
        experiment_names = list(experiments.keys())

    effective_steps = max_steps
    effective_early_stop = early_stop_pop

    results = []
    all_history = {}

    for name in experiment_names:
        if name not in experiments:
            print(f"Warning: experiment '{name}' not found in {experiments_file or DEFAULT_EXPERIMENTS_FILE}, skipping.")
            continue

        exp_config = experiments[name]
        desc = exp_config.get('description', '')
        desc_str = f' ({desc})' if desc else ''
        print(f"Running experiment: {name}{desc_str}...")

        exp = Experiment(name, exp_config['params'])
        exp.run(max_steps=effective_steps, early_stop_pop=effective_early_stop)
        summary = exp.summary()
        results.append(summary)
        all_history[name] = exp.results
        print(f"  Final pop: {summary['final_pop']}, Steps: {summary['steps_run']}")

    return results, all_history


def get_next_run_id(base_dir='results'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run_')]
    if not existing:
        return 1
    max_id = max(int(d.split('_')[1]) for d in existing)
    return max_id + 1


def save_results(results, all_history, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/experiment_summary.csv', index=False)
    print(f"\nSaved summary to {output_dir}/experiment_summary.csv")

    history_json = {}
    for name, history in all_history.items():
        history_json[name] = {}
        for k, v in history.items():
            history_json[name][k] = []
            for x in v:
                if isinstance(x, (np.integer,)):
                    history_json[name][k].append(int(x))
                elif isinstance(x, (np.floating,)):
                    history_json[name][k].append(float(x))
                else:
                    history_json[name][k].append(x)

    with open(f'{output_dir}/experiment_history.json', 'w') as f:
        json.dump(history_json, f)
    print(f"Saved history to {output_dir}/experiment_history.json")

    return df


def plot_results(df, all_history, output_dir='results'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, history in all_history.items():
        steps = history['step']
        pop = history['population']
        axes[0, 0].plot(steps, pop, label=name, linewidth=2)

    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_title('Population Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for name, history in all_history.items():
        steps = history['step']
        energy = history['avg_energy']
        axes[0, 1].plot(steps, energy, label=name, linewidth=2)

    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Average Energy')
    axes[0, 1].set_title('Average Energy Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for name, history in all_history.items():
        steps = history['step']
        sig_a = history['avg_signal_a']
        sig_b = history['avg_signal_b']
        axes[1, 0].plot(steps, sig_a, label=f'{name} A', linestyle='-', linewidth=1.5)
        axes[1, 0].plot(steps, sig_b, label=f'{name} B', linestyle='--', linewidth=1.5)

    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Signal Strength')
    axes[1, 0].set_title('Signal A (solid) vs B (dashed)')
    axes[1, 0].legend(loc='upper left', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    x = np.arange(len(df))
    width = 0.35
    axes[1, 1].bar(x - width/2, df['total_births'], width, label='Births', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, df['total_deaths'], width, label='Deaths', color='red', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df['name'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Total Births vs Deaths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/experiment_plots.png', dpi=150)
    print(f"Saved plots to {output_dir}/experiment_plots.png")
    plt.close()


def plot_trait_evolution(all_history, output_dir='results'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    trait_names = ['speed', 'sensing_radius', 'energy_efficiency']

    for idx, trait in enumerate(trait_names):
        for name, history in all_history.items():
            steps_with_traits = []
            trait_values = []
            for i, traits in enumerate(history['avg_traits']):
                if traits and trait in traits:
                    steps_with_traits.append(history['step'][i])
                    trait_values.append(traits[trait])

            if steps_with_traits:
                axes[idx].plot(steps_with_traits, trait_values, label=name, linewidth=2)

        axes[idx].set_xlabel('Step')
        axes[idx].set_ylabel(trait.replace('_', ' ').title())
        axes[idx].set_title(f'{trait.replace("_", " ").title()} Over Time')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/experiment_traits.png', dpi=150)
    print(f"Saved trait plots to {output_dir}/experiment_traits.png")
    plt.close()


def plot_table(df):
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    display_cols = ['name', 'final_pop', 'peak_pop', 'total_births', 'total_deaths', 'steps_run', 'avg_pop', 'final_energy']
    display_df = df[display_cols].copy()
    display_df['final_pop'] = display_df['final_pop'].astype(int)
    display_df['peak_pop'] = display_df['peak_pop'].astype(int)
    display_df['total_births'] = display_df['total_births'].astype(int)
    display_df['total_deaths'] = display_df['total_deaths'].astype(int)
    display_df['steps_run'] = display_df['steps_run'].astype(int)
    display_df['avg_pop'] = display_df['avg_pop'].round(1)
    display_df['final_energy'] = display_df['final_energy'].round(3)

    print(display_df.to_string(index=False))
    print("="*80)


def list_experiments(experiments_file=None):
    experiments, defaults = load_experiments(experiments_file)
    print(f"\nAvailable experiments (from {experiments_file or DEFAULT_EXPERIMENTS_FILE}):")
    print(f"  Defaults: max_steps={defaults.get('max_steps', 500)}, early_stop_pop={defaults.get('early_stop_pop', 5)}")
    print()
    for name, exp in experiments.items():
        desc = exp.get('description', 'No description')
        params = exp.get('params', {})
        print(f"  {name}: {desc}")
        for k, v in params.items():
            print(f"    {k}: {v}")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Nanocolony experiments')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='List of experiments to run (default: all in experiments.yml)')
    parser.add_argument('--steps', type=int, default=None, help='Max steps per experiment (overrides experiments.yml default)')
    parser.add_argument('--early-stop', type=int, default=None, help='Early stop population threshold')
    parser.add_argument('--file', type=str, default=None, help='Path to experiments YAML file')
    parser.add_argument('--list', action='store_true', help='List available experiments and exit')
    args = parser.parse_args()

    if args.list:
        list_experiments(args.file)
        exit(0)

    experiments, defaults = load_experiments(args.file)

    max_steps = args.steps if args.steps is not None else defaults.get('max_steps', 500)
    early_stop = args.early_stop if args.early_stop is not None else defaults.get('early_stop_pop', 5)

    print("="*60)
    print("NANOCOLONY EXPERIMENT SUITE")
    print("="*60)
    print(f"Config file: {args.file or DEFAULT_EXPERIMENTS_FILE}")
    print(f"Running experiments: {args.experiments or list(experiments.keys())}")
    print(f"Max steps per experiment: {max_steps}")
    print(f"Early stop population: {early_stop}")
    print("="*60)

    results, all_history = run_experiments(
        experiment_names=args.experiments,
        max_steps=max_steps,
        early_stop_pop=early_stop,
        experiments_file=args.file
    )

    run_id = get_next_run_id()
    output_dir = f'results/run_{run_id:03d}'

    df = save_results(results, all_history, output_dir)
    plot_results(df, all_history, output_dir)
    plot_trait_evolution(all_history, output_dir)
    plot_table(df)

    print(f"\nAll outputs saved to {output_dir}/")
