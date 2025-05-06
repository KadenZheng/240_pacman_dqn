#!/usr/bin/env python3
"""
Visualization script for Pac-Man RL experiments with different transition functions.

This script loads training metrics from CSV logs and experiment metrics from pickle files,
and generates plots to analyze agent performance across different transition function modes.
"""

import os
import csv
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def smooth(y, box_pts):
    """Apply a moving average filter to smooth the data."""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    # Padding the beginning to match original length
    padding = np.ones(box_pts - 1) * y_smooth[0]
    return np.concatenate((padding, y_smooth))

def plot_training_metrics(log_files):
    """
    Plot training metrics from CSV log files.
    
    Args:
        log_files: List of log file paths
    """
    if not log_files:
        print("No training log files found.")
        return
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Across Episodes', fontsize=16)
    
    # Create mapping for transition modes to readable labels
    mode_labels = {
        'fixed': 'Fixed Ghost Behavior',
        'progressive': 'Progressive Shifts',
        'domain_random': 'Domain Randomization',
        'random': 'Random Shifts',
    }
    
    # Create labels for each file by matching with saved model files
    file_labels = {}
    
    # Get all saved model files to match timestamps with transition modes
    saved_models = glob.glob('saves/model-*_final')
    model_info = {}
    
    # Extract transition modes from saved model filenames
    for model in saved_models:
        basename = os.path.basename(model)
        if '-' in basename:
            mode = basename.split('-')[1].split('_')[0]
            model_info[mode] = mode_labels.get(mode, mode.capitalize())
    
    # Match log files to transition modes
    timestamp_modes = {}
    
    # First, directly check if any transition mode is in the CSV content
    for f in log_files:
        timestamp = os.path.basename(f).split('-')[0]
        
        # Try to read the first few lines to find a transition mode reference
        transition_mode = None
        with open(f, 'r') as file:
            try:
                # Read a good chunk of the file to find transition mode info
                content = file.read(10000)  # Read first 10KB
                
                # Check for each transition mode in the content
                for mode in mode_labels.keys():
                    # Try different variants of how the mode might appear in the content
                    search_terms = [
                        f"transition_mode={mode}",
                        f"transition_mode: {mode}",
                        f"save_file: {mode}",
                        f"'save_file': '{mode}'",
                        f"save_file={mode}"
                    ]
                    if any(term in content for term in search_terms):
                        transition_mode = mode
                        break
                
                if transition_mode:
                    file_labels[f] = mode_labels[transition_mode]
                    timestamp_modes[timestamp] = transition_mode
            except:
                pass
    
    # For log files that we couldn't determine from content, 
    # try to infer based on timestamps and run order from the script
    if len(file_labels) < len(log_files):
        # Sort log files by timestamp
        sorted_logs = sorted(log_files, key=lambda x: os.path.basename(x).split('-')[0])
        
        # According to run_training_with_saving.sh, the order should be:
        # fixed, progressive, domain_random, random
        expected_order = ['fixed', 'progressive', 'domain_random', 'random']
        
        # Assign modes based on the expected order
        for i, f in enumerate(sorted_logs):
            if f not in file_labels and i < len(expected_order):
                mode = expected_order[i]
                file_labels[f] = mode_labels[mode]
    
    # Initialize data storage
    data = {
        'episode': defaultdict(list),
        'total_reward': defaultdict(list),
        'avg_q_value': defaultdict(list),
        'win': defaultdict(list),
        'steps': defaultdict(list),
        'epsilon': defaultdict(list)
    }
    
    # Load data from each file
    for f in log_files:
        with open(f, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                for key in data.keys():
                    try:
                        value = float(row[key])
                        data[key][f].append(value)
                    except (ValueError, KeyError):
                        # Skip if value can't be converted or key doesn't exist
                        pass
    
    # Plot total reward per episode
    ax = axes[0, 0]
    for f in log_files:
        x = data['episode'][f]
        y = data['total_reward'][f]
        if len(y) > 0:
            y_smooth = smooth(y, 50)  # Apply smoothing
            ax.plot(x, y_smooth, label=file_labels.get(f, os.path.basename(f).split('-')[0]))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Q-values per episode
    ax = axes[0, 1]
    for f in log_files:
        x = data['episode'][f]
        y = data['avg_q_value'][f]
        if len(y) > 0:
            y_smooth = smooth(y, 50)  # Apply smoothing
            ax.plot(x, y_smooth, label=file_labels.get(f, os.path.basename(f).split('-')[0]))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Q-value')
    ax.set_title('Q-values per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot win rate (moving average)
    ax = axes[1, 0]
    window_size = 100
    for f in log_files:
        x = data['episode'][f]
        y = data['win'][f]
        if len(y) > window_size:
            # Calculate moving average win rate
            win_rate = []
            for i in range(len(y) - window_size + 1):
                win_rate.append(sum(y[i:i+window_size]) / window_size)
            ax.plot(x[window_size-1:], win_rate, label=file_labels.get(f, os.path.basename(f).split('-')[0]))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (Moving Average)')
    ax.set_title(f'Win Rate ({window_size}-episode Window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot steps per episode
    ax = axes[1, 1]
    for f in log_files:
        x = data['episode'][f]
        y = data['steps'][f]
        if len(y) > 0:
            y_smooth = smooth(y, 50)  # Apply smoothing
            ax.plot(x, y_smooth, label=file_labels.get(f, os.path.basename(f).split('-')[0]))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('figures/training_metrics.png')
    plt.close()

def plot_experiment_results(metrics_files):
    """
    Plot experiment results from pickle files.
    
    Args:
        metrics_files: List of metrics file paths
    """
    if not metrics_files:
        print("No experiment metrics files found.")
        return
    
    # Mapping for more readable labels
    mode_display_names = {
        'fixed': 'Fixed Ghost Behavior',
        'progressive': 'Progressive Shifts',
        'domain_random': 'Domain Randomization',
        'random': 'Random Shifts',
    }
    
    # Load data from pickle files
    experiments = {}
    for f in metrics_files:
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                # Extract transition mode from filename
                filename = os.path.basename(f)
                transition_mode = filename.split('-')[-1].split('.')[0]
                # Use more descriptive name if available
                display_name = mode_display_names.get(transition_mode, transition_mode.capitalize())
                experiments[display_name] = data
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            print(f"Error loading {f}: {e}")
            continue
    
    if not experiments:
        print("No valid experiment data found.")
        return
    
    # Create a figure for experiment comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Comparison Across Transition Functions', fontsize=16)
    
    # Plot scores by transition mode
    ax = axes[0, 0]
    mode_labels = []
    scores = []
    score_stds = []
    
    for mode, data in experiments.items():
        if 'scores' in data and data['scores']:
            mode_labels.append(mode)
            scores.append(np.mean(data['scores']))
            score_stds.append(np.std(data['scores']))
    
    if mode_labels:
        bars = ax.bar(range(len(mode_labels)), scores, yerr=score_stds, 
                 capsize=10, alpha=0.7)
        ax.set_xticks(range(len(mode_labels)))
        ax.set_xticklabels(mode_labels, rotation=45)
        ax.set_ylabel('Average Score')
        ax.set_title('Average Score by Transition Mode')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + score_stds[i] + 5,
                   f'{scores[i]:.1f}', ha='center', va='bottom', rotation=0)
    
    # Plot win rates by transition mode
    ax = axes[0, 1]
    win_rates = []
    
    for mode, data in experiments.items():
        if 'wins' in data and data['wins']:
            win_rate = sum(data['wins']) / len(data['wins'])
            win_rates.append(win_rate * 100)  # Convert to percentage
    
    if mode_labels:
        bars = ax.bar(range(len(mode_labels)), win_rates, alpha=0.7)
        ax.set_xticks(range(len(mode_labels)))
        ax.set_xticklabels(mode_labels, rotation=45)
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Transition Mode')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{win_rates[i]:.1f}%', ha='center', va='bottom', rotation=0)
    
    # Plot average steps per episode by mode
    ax = axes[1, 0]
    steps = []
    steps_stds = []
    
    for mode, data in experiments.items():
        if 'steps' in data and data['steps']:
            steps.append(np.mean(data['steps']))
            steps_stds.append(np.std(data['steps']))
    
    if mode_labels:
        bars = ax.bar(range(len(mode_labels)), steps, yerr=steps_stds, 
                capsize=10, alpha=0.7)
        ax.set_xticks(range(len(mode_labels)))
        ax.set_xticklabels(mode_labels, rotation=45)
        ax.set_ylabel('Average Steps')
        ax.set_title('Average Steps per Episode by Transition Mode')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + steps_stds[i] + 5,
                   f'{steps[i]:.1f}', ha='center', va='bottom', rotation=0)
    
    # Plot ghost type distribution for domain randomization
    ax = axes[1, 1]
    domain_random_key = 'Domain Randomization'
    if domain_random_key in experiments and 'ghost_types' in experiments[domain_random_key]:
        ghost_types = experiments[domain_random_key]['ghost_types']
        type_counts = {}
        for ghost_type in ghost_types:
            if ghost_type in type_counts:
                type_counts[ghost_type] += 1
            else:
                type_counts[ghost_type] = 1
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Ghost Type Distribution in Domain Randomization')
    else:
        # If no domain_random data found, plot comparison of transition adaptability
        adaptability_scores = []
        for mode, data in experiments.items():
            if 'scores' in data and 'ghost_types' in data:
                # Group scores by ghost type
                ghost_scores = defaultdict(list)
                for i, ghost_type in enumerate(data['ghost_types']):
                    if i < len(data['scores']):
                        ghost_scores[ghost_type].append(data['scores'][i])
                
                # Calculate score variability across ghost types
                if len(ghost_scores) > 1:
                    # Calculate average score for each ghost type
                    avg_scores = [np.mean(scores) for scores in ghost_scores.values()]
                    # Lower standard deviation indicates better adaptability
                    adaptability = np.std(avg_scores)
                    adaptability_scores.append((mode, adaptability))
        
        if adaptability_scores:
            modes, values = zip(*adaptability_scores)
            # Normalize to make smaller values better (more adaptable)
            max_value = max(values)
            normalized = [1 - (v / max_value) for v in values]
            
            bars = ax.bar(range(len(modes)), normalized, alpha=0.7)
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45)
            ax.set_ylabel('Adaptability Score (normalized)')
            ax.set_title('Agent Adaptability Across Ghost Types')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('figures/experiment_results.png')
    plt.close()

def main():
    # Create output directory for plots if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Find all training log files
    print("Looking for training log files...")
    log_files = glob.glob(os.path.join('logs', '*-training-metrics.csv'))
    print(f"Found {len(log_files)} training log files.")
    
    # Plot training metrics
    plot_training_metrics(log_files)
    print("Generated training metrics plot in figures/training_metrics.png")
    
    # Find all experiment metrics files
    print("Looking for experiment metrics files...")
    metrics_files = glob.glob('overall-metrics-*.pkl')
    print(f"Found {len(metrics_files)} experiment metrics files.")
    
    # Plot experiment results
    plot_experiment_results(metrics_files)
    print("Generated experiment results plot in figures/experiment_results.png")

if __name__ == "__main__":
    main() 