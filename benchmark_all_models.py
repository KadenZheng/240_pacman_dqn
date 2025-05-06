#!/usr/bin/env python3
"""
benchmark_all_models.py
------------------
Script to benchmark all Pacman RL models against different ghost behaviors.
Generates comprehensive visualizations and comparison metrics.

Usage:
python benchmark_all_models.py --games 50 --layout mediumClassic --display

Models benchmarked:
- Fixed
- Progressive
- Domain Randomization
- Random
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from collections import defaultdict
import tensorflow as tf

# Import Pacman modules
from pacman import Directions, GameState, runGames
from pacmanDQN_Agents import PacmanDQN
from ghostAgents import TransitionFunctionController
from layout import getLayout
from game import Agent

# Directory for model checkpoints and results
MODELS_DIR = "saves"
RESULTS_DIR = "benchmark_results"

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

class BenchmarkAgent(Agent):
    """
    Base class for benchmark agents that wraps other agents
    and collects metrics during testing.
    """
    def __init__(self, agent, model_name):
        self.agent = agent
        self.model_name = model_name
        self.total_score = 0
        self.scores = []
        self.wins = 0
        self.losses = 0
        self.steps = []
        self.current_steps = 0
        self.q_values_history = []
        
    def getAction(self, state):
        self.current_steps += 1
        return self.agent.getAction(state)
    
    def registerInitialState(self, state):
        self.agent.registerInitialState(state)
        self.current_steps = 0
        
    def observationFunction(self, state):
        if hasattr(self.agent, 'observationFunction'):
            return self.agent.observationFunction(state)
        return state
    
    def final(self, state):
        self.total_score += state.getScore()
        self.scores.append(state.getScore())
        self.steps.append(self.current_steps)
        
        if state.isWin():
            self.wins += 1
        elif state.isLose():
            self.losses += 1
            
        if hasattr(self.agent, 'final'):
            self.agent.final(state)
    
    def get_metrics(self):
        total_games = self.wins + self.losses
        win_rate = self.wins / max(1, total_games)
        avg_score = np.mean(self.scores) if self.scores else 0
        avg_steps = np.mean(self.steps) if self.steps else 0
        
        return {
            'model': self.model_name,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'total_games': total_games,
            'wins': self.wins,
            'losses': self.losses,
            'scores': self.scores,
            'steps': self.steps
        }

class MonitoredPacmanDQN(PacmanDQN):
    """
    Extension of PacmanDQN that monitors Q-values for analysis
    """
    def __init__(self, args):
        super().__init__(args)
        self.q_value_history = []
        
    def getQValues(self, state):
        q_values = super().getQValues(state)
        # Record Q-values for analysis
        self.q_value_history.append({
            'avg': np.mean(q_values),
            'min': np.min(q_values),
            'max': np.max(q_values)
        })
        return q_values

def load_dqn_agent(model_path=None, model_name="unknown"):
    """
    Load a DQN agent with the specified model path.
    """
    layout = getLayout('mediumClassic')
    args = {
        'width': layout.width,
        'height': layout.height,
        'numTraining': 0  # No training during benchmarking
    }
    
    # Create a monitored DQN agent
    agent = MonitoredPacmanDQN(args)
    
    # Set the agent parameters
    agent.params['eps'] = 0.0  # No exploration during testing
    
    # Try to load the model if a path is specified
    if model_path and os.path.exists(model_path + '.index'):
        print(f"Loading {model_name} model from {model_path}")
        agent.qnet.sess.run(tf.compat.v1.global_variables_initializer())
        
        try:
            # Create a new saver with specific var_list to handle compatibility issues
            var_list = {}
            for var in tf.compat.v1.global_variables():
                # Skip optimizer variables that might be missing in the checkpoint
                if 'Adam' not in var.name and 'beta' not in var.name:
                    var_list[var.name.split(':')[0]] = var
            
            saver = tf.compat.v1.train.Saver(var_list=var_list)
            saver.restore(agent.qnet.sess, model_path)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Using untrained agent instead")
    else:
        print(f"No model found at {model_path}, using untrained agent")
    
    return agent

def generate_ghost_configs():
    """
    Generate configurations for different ghost behaviors to test against.
    """
    return [
        {'name': 'random', 'description': 'Random ghosts'},
        {'name': 'directional', 'description': 'Standard directional ghosts'},
        {'name': 'aggressive', 'description': 'Aggressive ghosts'},
        {'name': 'pathfinding', 'description': 'Pathfinding ghosts'},
        {'name': 'highly_random', 'description': 'Highly random ghosts'}
    ]

def create_ghosts(layout, ghost_behavior='directional'):
    """
    Create ghost agents with the specified behavior.
    """
    controller = TransitionFunctionController(mode='fixed')
    controller.current_ghost_type = ghost_behavior
    controller.reset_ghost_params()
    
    ghosts = []
    for i in range(layout.getNumGhosts()):
        ghosts.append(controller.create_ghost(i + 1))
    
    return ghosts

def benchmark_model(model_name, model_path, layout_name, num_games=100, display=None):
    """
    Benchmark a model against different ghost configurations.
    """
    # Create the agent
    agent = load_dqn_agent(model_path, model_name)
    benchmark_agent = BenchmarkAgent(agent, model_name)
    
    # Load the layout
    layout = getLayout(layout_name)
    
    # Get ghost configurations
    ghost_configs = generate_ghost_configs()
    
    # Store results
    results = []
    
    print(f"Starting benchmark for {model_name} model...")
    for ghost_config in ghost_configs:
        print(f"Testing against {ghost_config['description']}...")
        
        # Create ghosts
        ghosts = create_ghosts(layout, ghost_config['name'])
        
        # Run the games
        games = runGames(
            layout, 
            benchmark_agent, 
            ghosts, 
            display, 
            num_games, 
            record=False, 
            numTraining=0,
            catchExceptions=False
        )
        
        # Get metrics
        metrics = benchmark_agent.get_metrics()
        
        # Add to results
        results.append({
            'model': model_name,
            'ghost_behavior': ghost_config['name'],
            'ghost_description': ghost_config['description'],
            'win_rate': metrics['win_rate'],
            'avg_score': metrics['avg_score'],
            'avg_steps': metrics['avg_steps'],
            'wins': metrics['wins'],
            'losses': metrics['losses']
        })
        
        # Reset metrics for next ghost type
        benchmark_agent = BenchmarkAgent(agent, model_name)
    
    return results

def benchmark_all_models(layout_name, num_games=100, display=None):
    """
    Benchmark all models against different ghost configurations.
    """
    # Model names and paths
    models = [
        {'name': 'fixed', 'path': os.path.join(MODELS_DIR, 'model-fixed_final')},
        {'name': 'progressive', 'path': os.path.join(MODELS_DIR, 'model-progressive_final')},
        {'name': 'domain_random', 'path': os.path.join(MODELS_DIR, 'model-domain_random_final')},
        {'name': 'random', 'path': os.path.join(MODELS_DIR, 'model-random_final')}
    ]
    
    # Store all results
    all_results = []
    
    # Benchmark each model
    for model in models:
        results = benchmark_model(model['name'], model['path'], layout_name, num_games, display)
        all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV file
    benchmark_summary_path = os.path.join(RESULTS_DIR, 'all_models_benchmark_summary.csv')
    results_df.to_csv(benchmark_summary_path, index=False)
    print(f"Benchmark summary saved to {benchmark_summary_path}")
    
    return results_df

def generate_comparison_visualizations(benchmark_results):
    """
    Generate visualizations comparing the models.
    """
    # Set up the visualization style
    sns.set(style="whitegrid")
    
    # 1. Win Rate Comparison by Ghost Type
    plt.figure(figsize=(14, 8))
    sns.barplot(x='ghost_behavior', y='win_rate', hue='model', data=benchmark_results)
    plt.title('Win Rate by Ghost Type and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.legend(title='Model', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_comparison.png'))
    plt.close()
    
    # 2. Average Score Comparison by Ghost Type
    plt.figure(figsize=(14, 8))
    sns.barplot(x='ghost_behavior', y='avg_score', hue='model', data=benchmark_results)
    plt.title('Average Score by Ghost Type and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.legend(title='Model', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_score_comparison.png'))
    plt.close()
    
    # 3. Average Steps Comparison by Ghost Type
    plt.figure(figsize=(14, 8))
    sns.barplot(x='ghost_behavior', y='avg_steps', hue='model', data=benchmark_results)
    plt.title('Average Steps by Ghost Type and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Average Steps', fontsize=14)
    plt.legend(title='Model', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_steps_comparison.png'))
    plt.close()
    
    # 4. Overall Model Performance
    plt.figure(figsize=(12, 8))
    
    # Calculate average metrics for each model
    model_summary = benchmark_results.groupby('model').agg({
        'win_rate': 'mean',
        'avg_score': 'mean',
        'avg_steps': 'mean'
    }).reset_index()
    
    # Create subplot for win rate
    plt.subplot(1, 3, 1)
    sns.barplot(x='model', y='win_rate', data=model_summary)
    plt.title('Overall Win Rate', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Win Rate', fontsize=12)
    plt.ylim(0, 1)
    
    # Create subplot for average score
    plt.subplot(1, 3, 2)
    sns.barplot(x='model', y='avg_score', data=model_summary)
    plt.title('Overall Avg Score', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Average Score', fontsize=12)
    
    # Create subplot for average steps
    plt.subplot(1, 3, 3)
    sns.barplot(x='model', y='avg_steps', data=model_summary)
    plt.title('Overall Avg Steps', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Average Steps', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'overall_model_performance.png'))
    plt.close()
    
    # 5. Heatmap of Win Rates
    plt.figure(figsize=(12, 10))
    heatmap_data = benchmark_results.pivot(index='model', columns='ghost_behavior', values='win_rate')
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
    plt.title('Win Rate Heatmap by Model and Ghost Type', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_heatmap.png'))
    plt.close()
    
    # 6. Radar Chart for Model Comparison
    plt.figure(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Get unique ghost behaviors
    ghost_behaviors = benchmark_results['ghost_behavior'].unique()
    num_behaviors = len(ghost_behaviors)
    
    # Set up the angles for each behavior
    angles = np.linspace(0, 2*np.pi, num_behaviors, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Plot each model
    for model_name in benchmark_results['model'].unique():
        model_data = benchmark_results[benchmark_results['model'] == model_name]
        values = [model_data[model_data['ghost_behavior'] == behavior]['win_rate'].values[0] for behavior in ghost_behaviors]
        values += values[:1]  # Close the circle
        
        # Plot the model
        plt.plot(angles, values, linewidth=2, label=model_name)
        plt.fill(angles, values, alpha=0.1)
    
    # Set up the chart
    plt.xticks(angles[:-1], ghost_behaviors)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1.00'])
    plt.ylim(0, 1)
    plt.title('Win Rate by Ghost Type (Radar Chart)', fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_radar.png'))
    plt.close()
    
    print("Visualizations saved to benchmark_results directory")

def generate_summary_metrics(benchmark_results):
    """
    Generate summary metrics for all models.
    """
    # Calculate metrics for each model
    model_metrics = []
    
    for model_name in benchmark_results['model'].unique():
        model_data = benchmark_results[benchmark_results['model'] == model_name]
        
        # Calculate overall metrics
        overall_win_rate = model_data['win_rate'].mean()
        overall_score = model_data['avg_score'].mean()
        overall_steps = model_data['avg_steps'].mean()
        
        # Find best and worst ghost behaviors
        best_ghost = model_data.loc[model_data['win_rate'].idxmax()]
        worst_ghost = model_data.loc[model_data['win_rate'].idxmin()]
        
        metrics = {
            'model': model_name,
            'overall_win_rate': overall_win_rate,
            'overall_avg_score': overall_score,
            'overall_avg_steps': overall_steps,
            'best_ghost_behavior': best_ghost['ghost_behavior'],
            'best_ghost_win_rate': best_ghost['win_rate'],
            'worst_ghost_behavior': worst_ghost['ghost_behavior'],
            'worst_ghost_win_rate': worst_ghost['win_rate'],
            'win_rate_variance': model_data['win_rate'].var(),
            'total_wins': model_data['wins'].sum(),
            'total_losses': model_data['losses'].sum()
        }
        
        model_metrics.append(metrics)
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(model_metrics)
    metrics_path = os.path.join(RESULTS_DIR, 'model_summary_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Summary metrics saved to {metrics_path}")
    
    return metrics_df

def generate_benchmark_report(benchmark_results, summary_metrics):
    """
    Generate a markdown report summarizing benchmark results.
    """
    report_path = os.path.join(RESULTS_DIR, 'benchmark_comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Pacman RL Models Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares the performance of four different Pacman RL models:\n\n")
        f.write("1. **Fixed Model**: Trained on a fixed ghost behavior\n")
        f.write("2. **Progressive Model**: Trained with progressively changing ghost behaviors\n")
        f.write("3. **Domain Randomization Model**: Trained with randomized ghost behaviors\n")
        f.write("4. **Random Model**: Trained with completely random ghost behaviors\n\n")
        
        f.write("Each model was tested against five different ghost behaviors to evaluate its generalization capability.\n\n")
        
        f.write("## Model Performance Summary\n\n")
        
        # Format the summary metrics as a table
        f.write("| Model | Overall Win Rate | Avg Score | Avg Steps | Best Against | Worst Against |\n")
        f.write("|-------|-----------------|-----------|-----------|--------------|---------------|\n")
        
        for _, row in summary_metrics.iterrows():
            model = row['model']
            win_rate = f"{row['overall_win_rate']:.2f}"
            avg_score = f"{row['overall_avg_score']:.2f}"
            avg_steps = f"{row['overall_avg_steps']:.2f}"
            best = f"{row['best_ghost_behavior']} ({row['best_ghost_win_rate']:.2f})"
            worst = f"{row['worst_ghost_behavior']} ({row['worst_ghost_win_rate']:.2f})"
            
            f.write(f"| {model} | {win_rate} | {avg_score} | {avg_steps} | {best} | {worst} |\n")
        
        f.write("\n## Win Rate by Ghost Type\n\n")
        f.write("![Win Rate Comparison](win_rate_comparison.png)\n\n")
        
        f.write("The chart above shows each model's win rate against different ghost behaviors.\n\n")
        
        f.write("## Average Score by Ghost Type\n\n")
        f.write("![Average Score Comparison](avg_score_comparison.png)\n\n")
        
        f.write("## Overall Model Performance\n\n")
        f.write("![Overall Model Performance](overall_model_performance.png)\n\n")
        
        f.write("## Win Rate Heatmap\n\n")
        f.write("![Win Rate Heatmap](win_rate_heatmap.png)\n\n")
        
        f.write("The heatmap provides a clear visualization of each model's performance against each ghost type.\n\n")
        
        f.write("## Model Generalization (Radar Chart)\n\n")
        f.write("![Win Rate Radar](win_rate_radar.png)\n\n")
        
        f.write("The radar chart shows how well each model generalizes across different ghost behaviors.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Determine the best and worst overall models
        best_model = summary_metrics.loc[summary_metrics['overall_win_rate'].idxmax()]
        worst_model = summary_metrics.loc[summary_metrics['overall_win_rate'].idxmin()]
        
        # Model with least variance (most consistent)
        most_consistent = summary_metrics.loc[summary_metrics['win_rate_variance'].idxmin()]
        
        f.write(f"1. **Best Overall Model**: The {best_model['model']} model achieved the highest overall win rate of {best_model['overall_win_rate']:.2f}.\n")
        f.write(f"2. **Most Consistent Model**: The {most_consistent['model']} model showed the most consistent performance across different ghost behaviors with a variance of {most_consistent['win_rate_variance']:.4f}.\n")
        f.write(f"3. **Worst Overall Model**: The {worst_model['model']} model had the lowest overall win rate of {worst_model['overall_win_rate']:.2f}.\n\n")
        
        # Find which ghost type was most challenging overall
        ghost_difficulty = benchmark_results.groupby('ghost_behavior')['win_rate'].mean().reset_index()
        hardest_ghost = ghost_difficulty.loc[ghost_difficulty['win_rate'].idxmin()]
        easiest_ghost = ghost_difficulty.loc[ghost_difficulty['win_rate'].idxmax()]
        
        f.write(f"4. **Most Challenging Ghost**: {hardest_ghost['ghost_behavior']} ghosts were the most challenging across all models with an average win rate of {hardest_ghost['win_rate']:.2f}.\n")
        f.write(f"5. **Easiest Ghost**: {easiest_ghost['ghost_behavior']} ghosts were the easiest to handle with an average win rate of {easiest_ghost['win_rate']:.2f}.\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Draw conclusion based on the best model
        if best_model['model'] == 'domain_random' or best_model['model'] == 'progressive':
            f.write("The results suggest that training with variable ghost behaviors (either through domain randomization or progressive training) leads to better generalization across different ghost types compared to training with a fixed ghost behavior.\n\n")
        else:
            f.write("Interestingly, training with a fixed ghost behavior resulted in better overall performance than training with variable ghost behaviors. This might suggest that the fixed model developed a more robust strategy that coincidentally works well against different ghost types.\n\n")
        
        f.write("The performance differences across ghost types highlight the importance of considering various enemy behaviors when training and evaluating reinforcement learning agents in game environments.\n")
    
    print(f"Benchmark report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark all Pacman RL models against different ghost behaviors.')
    parser.add_argument('--games', type=int, default=50, help='Number of games to run for each benchmark')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout to use for benchmarking')
    parser.add_argument('--display', action='store_true', help='Whether to display the games graphically')
    
    args = parser.parse_args()
    
    # Set up display
    if args.display:
        from graphicsDisplay import PacmanGraphics
        display = PacmanGraphics(1.0)
    else:
        from textDisplay import NullGraphics
        display = NullGraphics()
    
    # Run benchmarks for all models
    benchmark_results = benchmark_all_models(args.layout, args.games, display)
    
    # Generate summary metrics
    summary_metrics = generate_summary_metrics(benchmark_results)
    
    # Generate visualizations
    generate_comparison_visualizations(benchmark_results)
    
    # Generate benchmark report
    generate_benchmark_report(benchmark_results, summary_metrics)
    
    # Print a simple summary
    print("\nOverall Benchmark Summary:")
    for model in summary_metrics['model'].unique():
        model_data = summary_metrics[summary_metrics['model'] == model]
        print(f"{model} model: Overall Win Rate = {model_data['overall_win_rate'].values[0]:.2f}")

if __name__ == "__main__":
    main() 