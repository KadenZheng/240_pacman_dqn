#!/usr/bin/env python3
"""
compare_all_models.py
------------------
Script to compare all three models (fixed, progressive, domain_random) 
based on their performance metrics.
Generates comprehensive visualizations and comparison statistics.
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
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
RESULTS_DIR = "comparison_results"

# Model names
MODEL_NAMES = ["fixed", "progressive", "domain_random"]

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        action = self.agent.getAction(state)
        
        # Record Q-values if agent has them and the history is not empty
        if hasattr(self.agent, 'q_value_history') and len(self.agent.q_value_history) > 0:
            self.q_values_history.append(self.agent.q_value_history[-1])
            
        return action
    
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
        
        # Q-value metrics
        if self.q_values_history:
            avg_q = np.mean([q['avg'] for q in self.q_values_history])
            min_q = np.mean([q['min'] for q in self.q_values_history])
            max_q = np.mean([q['max'] for q in self.q_values_history])
            q_range = np.mean([q['max'] - q['min'] for q in self.q_values_history])
        else:
            avg_q = min_q = max_q = q_range = 0
        
        return {
            'model': self.model_name,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'total_games': total_games,
            'wins': self.wins,
            'losses': self.losses,
            'scores': self.scores,
            'steps': self.steps,
            'avg_q_value': avg_q,
            'min_q_value': min_q,
            'max_q_value': max_q,
            'q_value_range': q_range
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

def load_dqn_agent(model_name, model_path=None):
    """
    Load a DQN agent with the specified model path.
    """
    layout = getLayout('mediumClassic')
    args = {
        'width': layout.width,
        'height': layout.height,
        'numTraining': 0,  # No training during benchmarking
        'transition_mode': model_name  # Set the transition mode for the agent
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

def benchmark_model(model_name, layout_name, num_games=100, display=None):
    """
    Benchmark a model against different ghost configurations.
    """
    # Find model in the saves directory
    model_path = os.path.join(MODELS_DIR, f'model-{model_name}_final')
    
    # Create the agent
    agent = load_dqn_agent(model_name, model_path)
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
        
        # Add ghost behavior info
        metrics['ghost_behavior'] = ghost_config['name']
        metrics['ghost_description'] = ghost_config['description']
        
        # Add to results
        results.append(metrics)
        
        # Reset metrics for next ghost type
        benchmark_agent = BenchmarkAgent(agent, model_name)
    
    return results

def compare_all_models(layout_name, num_games=100, display=None):
    """
    Compare all models against different ghost behaviors.
    """
    all_results = []
    
    for model_name in MODEL_NAMES:
        results = benchmark_model(model_name, layout_name, num_games, display)
        all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV file
    benchmark_summary_path = os.path.join(RESULTS_DIR, 'all_models_benchmark_summary.csv')
    results_df.to_csv(benchmark_summary_path, index=False)
    print(f"Benchmark summary saved to {benchmark_summary_path}")
    
    return results_df

def generate_comparison_visualizations(results_df):
    """
    Generate visualizations to compare model performance.
    """
    # 1. Win Rate Comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ghost_behavior', y='win_rate', hue='model', data=results_df, palette='viridis')
    plt.title('Win Rate by Ghost Behavior and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_comparison.png'))
    plt.close()
    
    # 2. Average Score Comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ghost_behavior', y='avg_score', hue='model', data=results_df, palette='viridis')
    plt.title('Average Score by Ghost Behavior and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_score_comparison.png'))
    plt.close()
    
    # 3. Average Steps (Survival Time) Comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ghost_behavior', y='avg_steps', hue='model', data=results_df, palette='viridis')
    plt.title('Average Steps per Game by Ghost Behavior and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Average Steps', fontsize=14)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_steps_comparison.png'))
    plt.close()
    
    # 4. Q-Value Comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ghost_behavior', y='avg_q_value', hue='model', data=results_df, palette='viridis')
    plt.title('Average Q-Value by Ghost Behavior and Model', fontsize=16)
    plt.xlabel('Ghost Behavior', fontsize=14)
    plt.ylabel('Average Q-Value', fontsize=14)
    plt.legend(title='Model Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'avg_q_value_comparison.png'))
    plt.close()
    
    # 5. Model Performance Radar Chart
    plt.figure(figsize=(14, 10))
    
    # Aggregate metrics by model
    model_metrics = results_df.groupby('model').agg({
        'win_rate': 'mean',
        'avg_score': 'mean',
        'avg_steps': 'mean',
        'avg_q_value': 'mean',
        'q_value_range': 'mean'
    }).reset_index()
    
    # Normalize metrics for radar chart
    for col in model_metrics.columns[1:]:
        model_metrics[col] = (model_metrics[col] - model_metrics[col].min()) / (model_metrics[col].max() - model_metrics[col].min())
    
    # Set up the radar chart
    categories = ['Win Rate', 'Avg Score', 'Avg Steps', 'Avg Q-Value', 'Q-Value Range']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw the y-axis labels (0 to 1)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], fontsize=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(model_metrics['model']):
        values = model_metrics.iloc[i, 1:].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), title="Model", fontsize=12)
    plt.title('Model Performance Comparison (Normalized)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_performance_radar.png'))
    plt.close()
    
    # 6. Score Distribution by Model
    plt.figure(figsize=(14, 8))
    
    # Prepare data for violin plot
    score_data = []
    for model in results_df['model'].unique():
        for _, row in results_df[results_df['model'] == model].iterrows():
            for score in row['scores']:
                score_data.append({'model': model, 'score': score})
    
    score_df = pd.DataFrame(score_data)
    
    # Create violin plot
    sns.violinplot(x='model', y='score', data=score_df, palette='viridis', inner='quartile')
    plt.title('Score Distribution by Model', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'score_distribution.png'))
    plt.close()
    
    # 7. Performance Heatmap
    # Pivot the data for the heatmap
    pivot_win_rate = results_df.pivot_table(
        values='win_rate', 
        index='model', 
        columns='ghost_behavior'
    )
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_win_rate, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Win Rate'})
    plt.title('Win Rate Heatmap: Model vs Ghost Behavior', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_heatmap.png'))
    plt.close()
    
    # 8. Model Robustness (Standard Deviation of Performance)
    # Calculate standard deviation of win rate and score across ghost behaviors
    robustness = results_df.groupby('model').agg({
        'win_rate': ['mean', 'std'],
        'avg_score': ['mean', 'std'],
    }).reset_index()
    
    robustness.columns = ['model', 'win_rate_mean', 'win_rate_std', 'score_mean', 'score_std']
    
    # Create a coefficient of variation (lower is better - more consistent)
    robustness['win_rate_cv'] = robustness['win_rate_std'] / robustness['win_rate_mean']
    robustness['score_cv'] = robustness['score_std'] / robustness['score_mean']
    
    # Plot robustness
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(robustness['model']))
    width = 0.35
    
    ax = plt.subplot(111)
    ax.bar(x - width/2, robustness['win_rate_cv'], width, label='Win Rate CV', color='steelblue')
    ax.bar(x + width/2, robustness['score_cv'], width, label='Score CV', color='darkorange')
    
    ax.set_xticks(x)
    ax.set_xticklabels(robustness['model'])
    
    ax.set_ylabel('Coefficient of Variation (lower is better)')
    ax.set_title('Model Robustness Across Ghost Behaviors')
    ax.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_robustness.png'))
    plt.close()

def generate_comparison_tables(results_df):
    """
    Generate detailed comparison tables.
    """
    # 1. Overall Performance Summary
    overall_summary = results_df.groupby('model').agg({
        'win_rate': 'mean',
        'avg_score': 'mean',
        'avg_steps': 'mean',
        'avg_q_value': 'mean',
        'q_value_range': 'mean'
    }).reset_index()
    
    # Add rank columns
    for col in overall_summary.columns[1:]:
        overall_summary[f'{col}_rank'] = overall_summary[col].rank(ascending=False)
    
    # Add a final average rank
    rank_cols = [col for col in overall_summary.columns if col.endswith('_rank')]
    overall_summary['avg_rank'] = overall_summary[rank_cols].mean(axis=1)
    
    # Sort by average rank
    overall_summary = overall_summary.sort_values('avg_rank')
    
    # Save to CSV
    overall_summary.to_csv(os.path.join(RESULTS_DIR, 'overall_performance_summary.csv'), index=False)
    
    # 2. Ghost-Specific Performance
    ghost_specific = results_df.pivot_table(
        values=['win_rate', 'avg_score'], 
        index='model',
        columns='ghost_behavior'
    )
    
    # Flatten multi-index columns
    ghost_specific.columns = [f'{col[1]}_{col[0]}' for col in ghost_specific.columns]
    ghost_specific = ghost_specific.reset_index()
    
    # Save to CSV
    ghost_specific.to_csv(os.path.join(RESULTS_DIR, 'ghost_specific_performance.csv'), index=False)
    
    # 3. Best Model by Ghost Type
    best_model_list = []
    for ghost in results_df['ghost_behavior'].unique():
        ghost_data = results_df[results_df['ghost_behavior'] == ghost]
        best_win_rate_idx = ghost_data['win_rate'].idxmax()
        best_score_idx = ghost_data['avg_score'].idxmax()
        
        best_model_list.append({
            'ghost_behavior': ghost,
            'best_win_rate_model': ghost_data.loc[best_win_rate_idx, 'model'],
            'best_win_rate': ghost_data.loc[best_win_rate_idx, 'win_rate'],
            'best_score_model': ghost_data.loc[best_score_idx, 'model'],
            'best_score': ghost_data.loc[best_score_idx, 'avg_score']
        })
    
    best_model = pd.DataFrame(best_model_list)
    
    # Save to CSV
    best_model.to_csv(os.path.join(RESULTS_DIR, 'best_model_by_ghost.csv'), index=False)
    
    return overall_summary, ghost_specific, best_model

def generate_comparison_report(results_df, overall_summary, ghost_specific, best_model):
    """
    Generate a detailed comparison report in markdown format.
    """
    report = f"""# Pacman DQN Model Comparison Report

## Overall Performance Summary

This report compares the performance of three different Pacman DQN models:
- **Fixed Model**: Trained on a fixed ghost behavior
- **Progressive Model**: Trained with progressively changing ghost behavior
- **Domain Randomization Model**: Trained with random ghost behavior

### Key Findings

The following table shows the overall performance metrics averaged across all ghost types:

| Model | Win Rate | Avg Score | Avg Steps | Avg Q-Value | Avg Rank |
|-------|----------|-----------|-----------|-------------|----------|
"""
    
    # Add rows for each model
    for _, row in overall_summary.iterrows():
        report += f"| {row['model']} | {row['win_rate']:.3f} | {row['avg_score']:.1f} | {row['avg_steps']:.1f} | {row['avg_q_value']:.3f} | {row['avg_rank']:.2f} |\n"
    
    report += """
## Performance by Ghost Behavior

Different models may perform better or worse depending on the ghost behavior they encounter.
The following sections analyze how each model performs against various ghost behaviors.

### Best Model for Each Ghost Type

| Ghost Behavior | Best Model (Win Rate) | Win Rate | Best Model (Score) | Avg Score |
|----------------|----------------------|----------|-------------------|-----------|
"""
    
    # Add rows for each ghost type
    for _, row in best_model.iterrows():
        report += f"| {row['ghost_behavior']} | {row['best_win_rate_model']} | {row['best_win_rate']:.3f} | {row['best_score_model']} | {row['best_score']:.1f} |\n"
    
    report += """
## Model Robustness

Robustness measures how consistently a model performs across different ghost behaviors.
A lower coefficient of variation indicates more consistent performance.

| Model | Win Rate CV | Score CV |
|-------|------------|----------|
"""
    
    # Calculate robustness metrics
    robustness = results_df.groupby('model').agg({
        'win_rate': ['mean', 'std'],
        'avg_score': ['mean', 'std'],
    })
    
    robustness.columns = ['win_rate_mean', 'win_rate_std', 'score_mean', 'score_std']
    robustness['win_rate_cv'] = robustness['win_rate_std'] / robustness['win_rate_mean']
    robustness['score_cv'] = robustness['score_std'] / robustness['score_mean']
    
    # Add rows for each model
    for model, row in robustness.iterrows():
        report += f"| {model} | {row['win_rate_cv']:.3f} | {row['score_cv']:.3f} |\n"
    
    report += """
## Conclusion

Based on the analysis, the following conclusions can be drawn:

"""
    
    # Add conclusion based on performance
    best_model_overall = overall_summary.iloc[0]['model']
    most_robust_model = robustness['win_rate_cv'].idxmin()
    
    report += f"1. **Best Overall Model**: {best_model_overall} has the highest average rank across all metrics.\n"
    report += f"2. **Most Consistent Model**: {most_robust_model} shows the most consistent performance across different ghost behaviors.\n"
    
    # Add specific ghost recommendations
    report += "3. **Ghost-Specific Recommendations**:\n"
    for _, row in best_model.iterrows():
        report += f"   - Against {row['ghost_behavior']} ghosts, use the {row['best_win_rate_model']} model for highest win rate.\n"
    
    # Save the report
    report_path = os.path.join(RESULTS_DIR, 'model_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to {report_path}")

def main():
    """
    Main function to run the model comparison.
    """
    parser = argparse.ArgumentParser(description='Compare all Pacman DQN models')
    parser.add_argument('--games', type=int, default=100, help='Number of games to run for each benchmark')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout to use for benchmarking')
    parser.add_argument('--display', action='store_true', help='Display the games graphically')
    args = parser.parse_args()
    
    # Set up display - FIX: Always use a display, either graphical or text-based
    if args.display:
        from graphicsDisplay import PacmanGraphics
        display = PacmanGraphics(1.0)
    else:
        from textDisplay import NullGraphics
        display = NullGraphics()
    
    # Run benchmarks for all models
    results_df = compare_all_models(args.layout, args.games, display)
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_comparison_visualizations(results_df)
    
    # Generate comparison tables
    print("Generating comparison tables...")
    overall_summary, ghost_specific, best_model = generate_comparison_tables(results_df)
    
    # Generate comparison report
    print("Generating comparison report...")
    generate_comparison_report(results_df, overall_summary, ghost_specific, best_model)
    
    print(f"All benchmarks and comparisons completed. Results are in the '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()