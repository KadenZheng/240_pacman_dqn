#!/usr/bin/env python3
"""
benchmark_fixed_model.py
------------------
Benchmarking script to test how the fixed model performs
when tested on unseen ghost behaviors.

This script:
1. Tests the fixed model against different ghost behaviors
2. Collects performance metrics
3. Generates visualizations and logs similar to analysis_results
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
LOGS_DIR = "logs"
RESULTS_DIR = "analysis_results"

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

class BenchmarkAgent(Agent):
    """
    Base class for benchmark agents that wraps other agents
    and collects metrics during testing.
    """
    def __init__(self, agent):
        self.agent = agent
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

def load_dqn_agent(model_path=None):
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
        print(f"Loading model from {model_path}")
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
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
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

def benchmark_fixed_model(layout_name, num_games=100, display=None):
    """
    Benchmark the fixed model against different ghost configurations.
    """
    # Find fixed model in the saves directory
    model_path = os.path.join(MODELS_DIR, 'model-fixed_final')
    
    # Create the agent
    agent = load_dqn_agent(model_path)
    benchmark_agent = BenchmarkAgent(agent)
    
    # Load the layout
    layout = getLayout(layout_name)
    
    # Get ghost configurations
    ghost_configs = generate_ghost_configs()
    
    # Store results
    results = []
    
    print("Starting benchmark for fixed model...")
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
            'ghost_behavior': ghost_config['name'],
            'ghost_description': ghost_config['description'],
            'win_rate': metrics['win_rate'],
            'avg_score': metrics['avg_score'],
            'avg_steps': metrics['avg_steps'],
            'wins': metrics['wins'],
            'losses': metrics['losses']
        })
        
        # Reset metrics for next ghost type
        benchmark_agent = BenchmarkAgent(agent)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV files
    benchmark_summary_path = os.path.join(RESULTS_DIR, 'benchmark_summary.csv')
    results_df.to_csv(benchmark_summary_path, index=False)
    print(f"Benchmark summary saved to {benchmark_summary_path}")
    
    return results_df, agent

def load_training_metrics():
    """
    Load training metrics from logs for analysis
    """
    training_logs = []
    for file in os.listdir(LOGS_DIR):
        if file.endswith('-training-metrics.csv'):
            log_path = os.path.join(LOGS_DIR, file)
            try:
                training_data = pd.read_csv(log_path)
                training_logs.append(training_data)
            except:
                print(f"Error reading log file: {log_path}")
    
    # If multiple log files, combine them (though we expect just one)
    if training_logs:
        return pd.concat(training_logs, ignore_index=True)
    return None

def create_performance_metrics(benchmark_results, training_data):
    """
    Create performance metrics similar to those in analysis_results
    """
    # Extract key metrics
    if training_data is not None:
        performance_metrics = {
            'metric': [
                'Average Final Win Rate',
                'Max Training Reward',
                'Final Training Reward',
                'Training Episodes',
                'Average Q-Value (Final)',
                'Training Time (hours)',
                'Benchmark Win Rate (Random Ghosts)',
                'Benchmark Win Rate (Directional Ghosts)',
                'Benchmark Win Rate (Aggressive Ghosts)'
            ],
            'value': [
                # Calculate from training data
                training_data.iloc[-100:]['win'].mean(),
                training_data['total_reward'].max(),
                training_data.iloc[-10:]['total_reward'].mean(),
                len(training_data),
                training_data.iloc[-10:]['avg_q_value'].mean(),
                len(training_data) * 0.5 / 60.0,  # Approximate training time
                
                # From benchmark results
                benchmark_results[benchmark_results['ghost_behavior'] == 'random']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'directional']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'aggressive']['win_rate'].values[0]
            ]
        }
    else:
        # If no training data, just use benchmark results
        performance_metrics = {
            'metric': [
                'Benchmark Win Rate (Random Ghosts)',
                'Benchmark Win Rate (Directional Ghosts)',
                'Benchmark Win Rate (Aggressive Ghosts)',
                'Benchmark Win Rate (Pathfinding Ghosts)',
                'Benchmark Win Rate (Highly Random Ghosts)'
            ],
            'value': [
                benchmark_results[benchmark_results['ghost_behavior'] == 'random']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'directional']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'aggressive']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'pathfinding']['win_rate'].values[0],
                benchmark_results[benchmark_results['ghost_behavior'] == 'highly_random']['win_rate'].values[0]
            ]
        }
    
    # Create DataFrame and save to CSV
    performance_df = pd.DataFrame(performance_metrics)
    performance_path = os.path.join(RESULTS_DIR, 'performance_metrics.csv')
    performance_df.to_csv(performance_path, index=False)
    print(f"Performance metrics saved to {performance_path}")
    
    return performance_df

def generate_visualizations(benchmark_results, training_data):
    """
    Generate visualizations similar to those in analysis_results
    """
    # Set up the visualization style
    sns.set(style="whitegrid")
    
    # 1. Benchmark Win Rate Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ghost_behavior', y='win_rate', data=benchmark_results)
    plt.title('Win Rate Across Ghost Behaviors')
    plt.xlabel('Ghost Behavior')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_comparison.png'))
    plt.close()
    
    if training_data is not None:
        # 2. Training Progression (similar to training_progression.png)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Total reward over time
        sns.lineplot(x='episode', y='total_reward', data=training_data, ax=ax1)
        ax1.set_title('Reward Progression During Training')
        ax1.set_ylabel('Total Reward')
        
        # Q-value progression
        sns.lineplot(x='episode', y='avg_q_value', data=training_data, ax=ax2)
        ax2.set_title('Average Q-Value Progression')
        ax2.set_ylabel('Average Q-Value')
        
        # Win rate (using a rolling window)
        training_data['win_rate'] = training_data['win'].rolling(window=50).mean()
        sns.lineplot(x='episode', y='win_rate', data=training_data, ax=ax3)
        ax3.set_title('Win Rate Progression (50-episode rolling average)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_progression.png'))
        plt.close()
        
        # 3. Q-value distribution (similar to q_value_distribution.png)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=training_data.iloc[-100:], x='avg_q_value', kde=True)
        plt.title('Q-Value Distribution (Last 100 Episodes)')
        plt.xlabel('Average Q-Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'q_value_distribution.png'))
        plt.close()
        
        # 4. Reward stability (similar to reward_stability.png)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='episode', y='total_reward', data=training_data)
        
        # Add a rolling average to show trend
        training_data['smooth_reward'] = training_data['total_reward'].rolling(window=50).mean()
        sns.lineplot(x='episode', y='smooth_reward', data=training_data, color='red')
        
        plt.title('Reward Stability During Training')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend(['Reward', '50-episode Moving Average'])
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'reward_stability.png'))
        plt.close()
        
        # 5. Correlation analysis (similar to correlation_analysis.png)
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix
        correlation_cols = ['total_reward', 'avg_q_value', 'win', 'steps', 'epsilon']
        corr_matrix = training_data[correlation_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Training Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'correlation_analysis.png'))
        plt.close()
    
    print("Visualizations saved to analysis_results directory")

def generate_analysis_report(benchmark_results, performance_metrics, training_data):
    """
    Generate a Markdown report similar to analysis_report.md
    """
    report_path = os.path.join(RESULTS_DIR, 'analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Pac-Man DQN Agent Analysis Report\n\n")
        
        f.write("## Benchmark Results\n\n")
        f.write("Performance across different ghost behaviors:\n\n")
        
        # Format benchmark results as a table
        f.write("| Ghost Type | Win Rate | Avg Score | Avg Steps |\n")
        f.write("|------------|----------|-----------|----------|\n")
        
        for _, row in benchmark_results.iterrows():
            f.write(f"| {row['ghost_description']} | {row['win_rate']:.2f} | {row['avg_score']:.2f} | {row['avg_steps']:.2f} |\n")
        
        f.write("\n## Performance Metrics\n\n")
        f.write("Key performance indicators:\n\n")
        
        # Format performance metrics as a table
        f.write("| Metric | Value |\n")
        f.write("|--------|------|\n")
        
        for _, row in performance_metrics.iterrows():
            metric_value = row['value']
            if isinstance(metric_value, float):
                formatted_value = f"{metric_value:.2f}"
            else:
                formatted_value = str(metric_value)
            
            f.write(f"| {row['metric']} | {formatted_value} |\n")
        
        if training_data is not None:
            f.write("\n## Training Analysis\n\n")
            f.write("The model was trained for {} episodes. The final average reward was {:.2f}.\n\n"
                    .format(len(training_data), training_data.iloc[-10:]['total_reward'].mean()))
            
            f.write("The visualizations in this directory show:\n\n")
            f.write("1. **Training Progression** - How reward, Q-values, and win rate evolved during training\n")
            f.write("2. **Q-Value Distribution** - Distribution of Q-values in the final stages of training\n")
            f.write("3. **Reward Stability** - How stable the rewards were throughout training\n")
            f.write("4. **Correlation Analysis** - Relationships between different training metrics\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("The fixed model demonstrates varying performance across different ghost behaviors. ")
        f.write("The model performs best against {} ghosts with a win rate of {:.2f}, ".format(
            benchmark_results.loc[benchmark_results['win_rate'].idxmax(), 'ghost_description'],
            benchmark_results['win_rate'].max()
        ))
        f.write("and worst against {} ghosts with a win rate of {:.2f}.\n".format(
            benchmark_results.loc[benchmark_results['win_rate'].idxmin(), 'ghost_description'],
            benchmark_results['win_rate'].min()
        ))
    
    print(f"Analysis report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark fixed Pacman model against different ghost behaviors.')
    parser.add_argument('--games', type=int, default=100, help='Number of games to run for each benchmark')
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
    
    # Run benchmark for fixed model
    benchmark_results, agent = benchmark_fixed_model(args.layout, args.games, display)
    
    # Load training metrics
    training_data = load_training_metrics()
    
    # Create performance metrics
    performance_metrics = create_performance_metrics(benchmark_results, training_data)
    
    # Generate visualizations
    generate_visualizations(benchmark_results, training_data)
    
    # Generate analysis report
    generate_analysis_report(benchmark_results, performance_metrics, training_data)
    
    # Print summary
    print("\nBenchmark Summary for Fixed Model:")
    for _, row in benchmark_results.iterrows():
        print(f"{row['ghost_description']}: Win Rate = {row['win_rate']:.2f}, Avg Score = {row['avg_score']:.2f}")

if __name__ == "__main__":
    main() 