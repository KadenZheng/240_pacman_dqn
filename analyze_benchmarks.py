#!/usr/bin/env python3
"""
Benchmark analysis script for Pac-Man RL experiments.

This script analyzes the results of benchmark tests across different agent types
and transition functions to understand how well each agent generalizes.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_benchmark_file(filepath):
    """
    Parse a benchmark output file to extract scores and win information.
    
    Args:
        filepath: Path to the benchmark file
        
    Returns:
        dict: Dictionary with scores, wins, and other metrics
    """
    results = {
        'scores': [],
        'wins': [],
        'num_games': 0
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract score information
            score_pattern = r'Average Score:\s+([-\d.]+)'
            score_match = re.search(score_pattern, content)
            if score_match:
                avg_score = float(score_match.group(1))
                
                # Extract individual scores
                scores_pattern = r'Scores:\s+([-\d.,\s]+)'
                scores_match = re.search(scores_pattern, content)
                if scores_match:
                    scores_str = scores_match.group(1).strip()
                    scores = [float(s) for s in scores_str.split(',')]
                    results['scores'] = scores
                    results['num_games'] = len(scores)
            
            # Extract win rate information
            win_pattern = r'Win Rate:\s+(\d+)/(\d+)\s+\(([-\d.]+)\)'
            win_match = re.search(win_pattern, content)
            if win_match:
                wins = int(win_match.group(1))
                total = int(win_match.group(2))
                win_rate = float(win_match.group(3))
                
                # Create a list of 1s (win) and 0s (loss)
                results['wins'] = [1 if w == 'Win' else 0 for w in re.findall(r'(Win|Loss)', content)]
                
                # Double-check the win count
                if len(results['wins']) > 0:
                    if sum(results['wins']) != wins:
                        print(f"Warning: Win count mismatch in {filepath}")
                        results['wins'] = [1] * wins + [0] * (total - wins)
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return results

def calculate_metrics(results):
    """
    Calculate additional metrics from benchmark results.
    
    Args:
        results: Dictionary with scores and wins
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    metrics = {}
    
    if 'scores' in results and results['scores']:
        metrics['avg_score'] = np.mean(results['scores'])
        metrics['std_score'] = np.std(results['scores'])
        metrics['max_score'] = np.max(results['scores'])
        metrics['min_score'] = np.min(results['scores'])
    
    if 'wins' in results and results['wins']:
        metrics['win_rate'] = sum(results['wins']) / len(results['wins'])
    
    return metrics

def plot_benchmark_comparison(data, output_file='benchmark_comparison.png'):
    """
    Create a comparison plot of benchmark results.
    
    Args:
        data: Dictionary with benchmark data for each agent/mode
        output_file: Path to save the output plot
    """
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Benchmark Comparison Across Agents and Transition Functions', fontsize=16)
    
    # Group benchmarks by test scenario
    scenarios = defaultdict(dict)
    
    for key, results in data.items():
        # Parse key to get agent type and scenario
        parts = key.split('_')
        
        if 'test' in key:
            # Test scenarios (e.g., test_aggressive_fixed)
            if 'aggressive' in key:
                scenario = 'aggressive'
            elif 'random' in key:
                scenario = 'random'
            else:
                scenario = 'unknown'
                
            if 'fixed' in key:
                agent = 'fixed'
            elif 'domain' in key:
                agent = 'domain_random'
            elif 'heuristic' in key:
                agent = 'heuristic'
            else:
                agent = 'unknown'
        else:
            # Standard benchmarks
            scenario = 'standard'
            if 'fixed' in key:
                agent = 'fixed'
            elif 'progressive' in key:
                agent = 'progressive'
            elif 'domain' in key:
                agent = 'domain_random'
            elif 'heuristic' in key:
                agent = 'heuristic'
            else:
                agent = 'unknown'
        
        scenarios[scenario][agent] = results
    
    # Plot average scores by scenario
    ax = axes[0, 0]
    plot_grouped_bars(ax, scenarios, 'avg_score', 'Average Score', 'Average Score by Scenario')
    
    # Plot win rates by scenario
    ax = axes[0, 1]
    plot_grouped_bars(ax, scenarios, 'win_rate', 'Win Rate', 'Win Rate by Scenario', percentage=True)
    
    # Plot standard deviation of scores (lower is more consistent)
    ax = axes[1, 0]
    plot_grouped_bars(ax, scenarios, 'std_score', 'Score Std. Dev.', 'Score Variability by Scenario')
    
    # Plot score distribution for standard scenario
    ax = axes[1, 1]
    if 'standard' in scenarios:
        for agent, results in scenarios['standard'].items():
            if 'scores' in results:
                ax.hist(results['scores'], alpha=0.5, label=agent)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution (Standard Scenario)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file)
    plt.close()

def plot_grouped_bars(ax, scenarios, metric, ylabel, title, percentage=False):
    """
    Create a grouped bar chart for a specific metric.
    
    Args:
        ax: Matplotlib axis
        scenarios: Dictionary with scenario data
        metric: Metric to plot
        ylabel: Y-axis label
        title: Plot title
        percentage: Whether to display as percentage
    """
    # Set up bar positions
    scenario_names = list(scenarios.keys())
    agent_names = set()
    for scenario_data in scenarios.values():
        agent_names.update(scenario_data.keys())
    agent_names = sorted(list(agent_names))
    
    width = 0.8 / len(agent_names)
    
    # Plot bars for each agent within each scenario
    for i, agent in enumerate(agent_names):
        values = []
        positions = []
        
        for j, scenario in enumerate(scenario_names):
            if agent in scenarios[scenario] and metric in scenarios[scenario][agent]:
                values.append(scenarios[scenario][agent][metric])
                positions.append(j + (i - len(agent_names)/2 + 0.5) * width)
        
        # Convert to percentage if needed
        if percentage:
            values = [v * 100 for v in values]
            
        # Create bars
        bars = ax.bar(positions, values, width, label=agent, alpha=0.7)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                  f'{value:.1f}{"%" if percentage else ""}', 
                  ha='center', va='bottom', rotation=0)
    
    # Set labels and title
    ax.set_xlabel('Scenario')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_adaptability_comparison(data, output_file='adaptability_comparison.png'):
    """
    Create a special plot focusing on adaptability across different ghost behaviors.
    
    Args:
        data: Dictionary with benchmark data for each agent/mode
        output_file: Path to save the output plot
    """
    # Group results by agent type
    agent_performance = defaultdict(dict)
    
    # Scenarios to analyze
    scenarios = ['standard', 'aggressive', 'random']
    
    # Extract relevant agents
    agents = ['fixed', 'domain_random', 'heuristic']
    
    # Collect performance data for each agent across scenarios
    for key, results in data.items():
        for scenario in scenarios:
            if scenario in key:
                for agent in agents:
                    if agent in key:
                        # Store score and win rate for this agent in this scenario
                        if 'metrics' in results:
                            agent_performance[agent][scenario] = {
                                'score': results['metrics']['avg_score'],
                                'win_rate': results['metrics']['win_rate']
                            }
    
    # Calculate adaptability metrics
    adaptability = {}
    for agent in agent_performance:
        if len(agent_performance[agent]) >= 2:  # Need at least 2 scenarios
            # Calculate how well score is maintained across scenarios
            if 'standard' in agent_performance[agent]:
                standard_score = agent_performance[agent]['standard'].get('score', 0)
                scores = []
                for scenario in agent_performance[agent]:
                    if scenario != 'standard':
                        scenario_score = agent_performance[agent][scenario].get('score', 0)
                        scores.append(scenario_score / max(1, standard_score))  # Normalized score
                
                if scores:
                    # Adaptability is the average normalized score across non-standard scenarios
                    adaptability[agent] = sum(scores) / len(scores)
    
    # Create the adaptability plot
    plt.figure(figsize=(10, 6))
    
    # Plot adaptability scores
    agents = list(adaptability.keys())
    scores = [adaptability[agent] for agent in agents]
    
    bars = plt.bar(agents, scores, alpha=0.7)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{score:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Agent Type')
    plt.ylabel('Adaptability Score')
    plt.title('Agent Adaptability Across Different Ghost Behaviors')
    plt.ylim(0, max(scores) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    """Main function to analyze benchmark results"""
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    benchmark_dir = 'logs/benchmarks'
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory {benchmark_dir} not found.")
        return
    
    # Find all benchmark files
    benchmark_files = [os.path.join(benchmark_dir, f) for f in os.listdir(benchmark_dir) 
                      if f.endswith('.txt')]
    
    if not benchmark_files:
        print("No benchmark files found.")
        return
    
    # Parse each benchmark file
    benchmark_data = {}
    for filepath in benchmark_files:
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        
        print(f"Parsing {filename}...")
        results = parse_benchmark_file(filepath)
        
        if results['num_games'] > 0:
            metrics = calculate_metrics(results)
            benchmark_data[name] = {
                'scores': results['scores'],
                'wins': results['wins'],
                'metrics': metrics
            }
            print(f"  Average Score: {metrics.get('avg_score', 'N/A')}")
            print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
        else:
            print(f"  No valid data found.")
    
    # Plot benchmark comparison
    plot_benchmark_comparison(benchmark_data, 'plots/benchmark_comparison.png')
    print("Generated benchmark comparison plot.")
    
    # Plot adaptability comparison
    plot_adaptability_comparison(benchmark_data, 'plots/adaptability_comparison.png')
    print("Generated adaptability comparison plot.")
    
    # Save a summary report
    with open('benchmark_summary.txt', 'w') as f:
        f.write("Benchmark Summary\n")
        f.write("================\n\n")
        
        for name, data in benchmark_data.items():
            metrics = data['metrics']
            f.write(f"Agent: {name}\n")
            f.write(f"  Average Score: {metrics.get('avg_score', 'N/A'):.2f}\n")
            f.write(f"  Score Std Dev: {metrics.get('std_score', 'N/A'):.2f}\n")
            f.write(f"  Win Rate: {metrics.get('win_rate', 'N/A') * 100:.1f}%\n")
            f.write(f"  Games: {len(data['scores'])}\n")
            f.write("\n")
    
    print("Saved benchmark summary to benchmark_summary.txt")

if __name__ == "__main__":
    main() 