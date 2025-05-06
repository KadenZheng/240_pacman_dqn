#!/usr/bin/env python3
"""
training_log_analyzer.py
-----------------------
Script to analyze and compare training logs from different Pacman RL model training approaches.
Generates comprehensive visualizations and comparison statistics for training metrics.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Directory for results
RESULTS_DIR = "v1_results"

# Ensure directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_training_logs(file_paths):
    """
    Load training logs from CSV files.
    
    Args:
        file_paths: Dictionary mapping model names to file paths
        
    Returns:
        Dictionary mapping model names to pandas DataFrames
    """
    training_data = {}
    for model_name, file_path in file_paths.items():
        try:
            df = pd.read_csv(file_path)
            # Add model name as a column for easier comparison
            df['model'] = model_name
            training_data[model_name] = df
            print(f"Loaded {model_name} training data with {len(df)} entries")
        except Exception as e:
            print(f"Error loading {model_name} training data: {e}")
    
    return training_data

def preprocess_training_data(training_data):
    """
    Preprocess training data for analysis.
    
    Args:
        training_data: Dictionary mapping model names to DataFrames
        
    Returns:
        Combined DataFrame with all training data
    """
    # Combine all data into a single DataFrame
    all_data = pd.concat(training_data.values(), ignore_index=True)
    
    # Handle common preprocessing tasks
    # Convert timestamps if needed, calculate cumulative metrics, etc.
    
    return all_data

def normalize_episode_count(training_data):
    """
    Normalize the episode count across all models to ensure fair comparison.
    
    Args:
        training_data: Dictionary mapping model names to DataFrames
        
    Returns:
        Dictionary with normalized DataFrames
    """
    min_episodes = min(len(df) for df in training_data.values())
    print(f"Normalizing all datasets to {min_episodes} episodes for fair comparison")
    
    normalized_data = {}
    for model_name, df in training_data.items():
        normalized_data[model_name] = df.iloc[:min_episodes].copy()
    
    return normalized_data

def generate_rolling_window_plot(individual_dfs, column, title, ylabel, filename, window_size_factor=10):
    """
    Generate a plot with rolling window average for a specific column.
    
    Args:
        individual_dfs: Dictionary of DataFrames
        column: Column name to plot
        title: Plot title
        ylabel: Y-axis label
        filename: Output filename
        window_size_factor: Factor to determine rolling window size
    """
    plt.figure(figsize=(14, 8))
    
    for model, df in individual_dfs.items():
        if column in df.columns:
            # Use rolling average to smooth the curve
            window_size = min(100, len(df) // window_size_factor) if len(df) > 100 else 10
            df[f'rolling_{column}'] = df[column].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['episode'], df[f'rolling_{column}'], label=f"{model}")
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def generate_learning_curve_plots(all_data, individual_dfs):
    """
    Generate learning curve plots comparing the models.
    """
    # Plot total_reward over episodes
    if 'episode' in all_data.columns and 'total_reward' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'total_reward', 
            'Total Reward Over Episodes', 
            'Average Total Reward (Rolling Window)', 
            'learning_curve_total_reward.png'
        )
    
    # Plot win rate over episodes
    if 'episode' in all_data.columns and 'win' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'win', 
            'Win Rate Over Episodes', 
            'Win Rate (Rolling Window)', 
            'learning_curve_win_rate.png'
        )
    
    # Plot steps per episode
    if 'episode' in all_data.columns and 'steps' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'steps', 
            'Steps Per Episode', 
            'Average Steps (Rolling Window)', 
            'learning_curve_steps.png'
        )

def generate_q_value_analysis(all_data, individual_dfs):
    """
    Generate Q-value analysis plots.
    """
    # Plot average Q-value over episodes
    if 'episode' in all_data.columns and 'avg_q_value' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'avg_q_value', 
            'Average Q-Value Over Episodes', 
            'Average Q-Value (Rolling Window)', 
            'q_value_average.png'
        )
    
    # Plot min Q-value over episodes
    if 'episode' in all_data.columns and 'min_q_value' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'min_q_value', 
            'Minimum Q-Value Over Episodes', 
            'Minimum Q-Value (Rolling Window)', 
            'q_value_minimum.png'
        )
    
    # Plot max Q-value over episodes
    if 'episode' in all_data.columns and 'max_q_value' in all_data.columns:
        generate_rolling_window_plot(
            individual_dfs, 
            'max_q_value', 
            'Maximum Q-Value Over Episodes', 
            'Maximum Q-Value (Rolling Window)', 
            'q_value_maximum.png'
        )
    
    # Plot Q-value range (max - min)
    if 'min_q_value' in all_data.columns and 'max_q_value' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            window_size = min(100, len(df) // 10) if len(df) > 100 else 10
            df['q_value_range'] = df['max_q_value'] - df['min_q_value']
            df['rolling_q_range'] = df['q_value_range'].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['episode'], df['rolling_q_range'], label=f"{model}")
            
            # Also plot the range as a shaded area
            if model == list(individual_dfs.keys())[0]:  # Only do this for the first model to avoid clutter
                plt.fill_between(
                    df['episode'],
                    df['min_q_value'].rolling(window=window_size, min_periods=1).mean(),
                    df['max_q_value'].rolling(window=window_size, min_periods=1).mean(),
                    alpha=0.1,
                    label=f"{model} Q-Range"
                )
        
        plt.title('Q-Value Range Over Episodes', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Q-Value Range (Rolling Window)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'q_value_range.png'))
        plt.close()

def generate_exploration_analysis(all_data, individual_dfs):
    """
    Generate exploration analysis plots.
    """
    # Plot epsilon (exploration rate) over episodes
    if 'episode' in all_data.columns and 'epsilon' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            plt.plot(df['episode'], df['epsilon'], label=f"{model}")
        
        plt.title('Exploration Rate (Epsilon) Over Episodes', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Epsilon', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'exploration_rate.png'))
        plt.close()
    
    # Plot steps vs epsilon correlation
    if 'steps' in all_data.columns and 'epsilon' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            plt.scatter(df['epsilon'], df['steps'], alpha=0.3, label=f"{model}")
            
            # Add trend line
            z = np.polyfit(df['epsilon'], df['steps'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df['epsilon']), p(sorted(df['epsilon'])), linestyle='--')
        
        plt.title('Steps vs. Exploration Rate (Epsilon)', fontsize=16)
        plt.xlabel('Epsilon', fontsize=14)
        plt.ylabel('Steps', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'steps_vs_epsilon.png'))
        plt.close()

def generate_performance_distribution_plots(all_data, individual_dfs):
    """
    Generate distribution plots for key performance metrics.
    """
    performance_metrics = ['total_reward', 'steps', 'win']
    
    for metric in performance_metrics:
        if metric in all_data.columns:
            plt.figure(figsize=(14, 8))
            
            for model, df in individual_dfs.items():
                sns.kdeplot(df[metric], label=f"{model}", fill=True, alpha=0.3)
            
            plt.title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=16)
            plt.xlabel(metric.replace('_', ' ').title(), fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'distribution_{metric}.png'))
            plt.close()

def generate_violin_plots(all_data):
    """
    Generate violin plots for comparing distributions across models.
    """
    numeric_columns = all_data.select_dtypes(include=np.number).columns.tolist()
    # Exclude episode and other non-performance columns
    plot_columns = [col for col in numeric_columns if col not in ['episode', 'model', 'epsilon']]
    
    for col in plot_columns:
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='model', y=col, data=all_data, palette='viridis')
        plt.title(f'Distribution of {col.replace("_", " ").title()} by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(col.replace('_', ' ').title(), fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'violin_{col}.png'))
        plt.close()

def generate_correlation_heatmaps(all_data, individual_dfs):
    """
    Generate correlation heatmaps for each model.
    """
    numeric_columns = all_data.select_dtypes(include=np.number).columns.tolist()
    # Exclude non-performance columns
    corr_columns = [col for col in numeric_columns if col not in ['model']]
    
    # Overall correlation
    plt.figure(figsize=(12, 10))
    corr_matrix = all_data[corr_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Overall Correlation Between Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_overall.png'))
    plt.close()
    
    # Per-model correlation
    for model, df in individual_dfs.items():
        plt.figure(figsize=(12, 10))
        corr_matrix = df[corr_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f'Correlation Between Metrics - {model}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'correlation_{model}.png'))
        plt.close()

def generate_learning_efficiency_plot(all_data, individual_dfs):
    """
    Generate plots showing learning efficiency (reward per episode).
    """
    if 'total_reward' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            # Calculate cumulative reward
            df['cumulative_reward'] = df['total_reward'].cumsum()
            plt.plot(df['episode'], df['cumulative_reward'], label=f"{model}")
        
        plt.title('Cumulative Reward Over Episodes', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Cumulative Reward', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'cumulative_reward.png'))
        plt.close()
        
        # Calculate moving average of reward gained per 100 episodes
        window_size = 100
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            df['reward_rate'] = df['total_reward'].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['episode'], df['reward_rate'], label=f"{model}")
        
        plt.title(f'Learning Efficiency (Avg Reward per {window_size} Episodes)', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel(f'Average Reward (Window={window_size})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'learning_efficiency.png'))
        plt.close()

def generate_milestone_comparison(all_data, individual_dfs):
    """
    Generate comparison of how quickly models reach certain performance milestones.
    """
    if 'total_reward' in all_data.columns:
        # Define milestones as percentiles of the overall reward distribution
        reward_percentiles = [25, 50, 75, 90]
        reward_milestones = [np.percentile(all_data['total_reward'], p) for p in reward_percentiles]
        
        milestone_data = []
        
        for model, df in individual_dfs.items():
            model_milestones = {}
            model_milestones['model'] = model
            
            for percentile, threshold in zip(reward_percentiles, reward_milestones):
                # Find the first episode where reward exceeds the threshold
                episodes_to_milestone = df[df['total_reward'] >= threshold]['episode'].min()
                model_milestones[f'episodes_to_{percentile}th_percentile'] = episodes_to_milestone
            
            milestone_data.append(model_milestones)
        
        milestone_df = pd.DataFrame(milestone_data)
        
        # Create a bar chart
        plt.figure(figsize=(14, 8))
        milestone_cols = [col for col in milestone_df.columns if col.startswith('episodes_to_')]
        
        for i, col in enumerate(milestone_cols):
            percentile = col.split('_')[2].replace('th', '')
            plt.bar(
                np.arange(len(milestone_df)) + (i * 0.2), 
                milestone_df[col], 
                width=0.2,
                label=f'{percentile}th Percentile'
            )
        
        plt.xticks(np.arange(len(milestone_df)) + 0.3, milestone_df['model'])
        plt.title('Episodes to Reach Reward Milestones', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Episodes', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'milestone_comparison.png'))
        plt.close()
        
        # Save the milestone data
        milestone_df.to_csv(os.path.join(RESULTS_DIR, 'milestone_data.csv'), index=False)

def generate_win_percentage_over_time(all_data, individual_dfs):
    """
    Generate plot showing win percentage over time with confidence intervals.
    """
    if 'win' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            # Calculate win percentage in chunks of episodes
            chunk_size = max(1, len(df) // 20)  # Divide into 20 chunks
            win_pcts = []
            episodes = []
            errors = []
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                win_pct = chunk['win'].mean() * 100
                win_pcts.append(win_pct)
                episodes.append(chunk['episode'].mean())
                
                # Calculate 95% confidence interval
                n = len(chunk)
                std_err = np.sqrt((win_pct/100 * (1 - win_pct/100)) / n) * 100
                margin_error = 1.96 * std_err  # 95% confidence
                errors.append(margin_error)
            
            plt.errorbar(episodes, win_pcts, yerr=errors, label=f"{model}", marker='o', capsize=5)
        
        plt.title('Win Percentage Over Training', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Win Percentage (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'win_percentage_over_time.png'))
        plt.close()

def generate_training_stability_analysis(all_data, individual_dfs):
    """
    Generate analysis of training stability.
    """
    # Plot training stability for total reward
    if 'total_reward' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            window_size = min(100, len(df) // 10) if len(df) > 100 else 10
            df['reward_variance'] = df['total_reward'].rolling(window=window_size, min_periods=1).var()
            plt.plot(df['episode'], df['reward_variance'], label=f"{model}")
        
        plt.title('Training Stability (Reward Variance) Over Episodes', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward Variance (Rolling Window)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'reward_stability.png'))
        plt.close()
    
    # Plot training stability for q-values
    if 'avg_q_value' in all_data.columns:
        plt.figure(figsize=(14, 8))
        
        for model, df in individual_dfs.items():
            window_size = min(100, len(df) // 10) if len(df) > 100 else 10
            df['q_variance'] = df['avg_q_value'].rolling(window=window_size, min_periods=1).var()
            plt.plot(df['episode'], df['q_variance'], label=f"{model}")
        
        plt.title('Training Stability (Q-Value Variance) Over Episodes', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Q-Value Variance (Rolling Window)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'q_value_stability.png'))
        plt.close()

def generate_comparison_metrics(all_data, individual_dfs):
    """
    Generate summary metrics comparing the models.
    """
    # Calculate key metrics for each model
    metrics = []
    
    for model, df in individual_dfs.items():
        # Final performance (last 10% of training)
        final_df = df.iloc[int(0.9 * len(df)):]
        
        metric = {
            'model': model,
            'episodes': len(df),
            'final_avg_reward': final_df['total_reward'].mean() if 'total_reward' in df.columns else None,
            'final_win_rate': final_df['win'].mean() if 'win' in df.columns else None,
            'final_avg_steps': final_df['steps'].mean() if 'steps' in df.columns else None,
            'training_time': (df['episode'].max() - df['episode'].min() + 1),  # Approximating with episode count
            'reward_improvement': df['total_reward'].iloc[-100:].mean() - df['total_reward'].iloc[:100].mean() if 'total_reward' in df.columns and len(df) > 200 else None,
        }
        
        # Add Q-value metrics if available
        for col in df.columns:
            if 'q_value' in col.lower() and col != 'q_value_range':
                metric[f'final_{col}'] = final_df[col].mean()
        
        metrics.append(metric)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Save to CSV
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'training_comparison_metrics.csv'), index=False)
    
    # Create a radar chart of normalized metrics
    if not metrics_df.empty:
        plt.figure(figsize=(10, 10))
        
        # Select numerical columns for the radar chart
        radar_cols = [col for col in metrics_df.columns if col != 'model' and pd.api.types.is_numeric_dtype(metrics_df[col]) and not metrics_df[col].isnull().all()]
        
        # Normalize the metrics
        for col in radar_cols:
            if not metrics_df[col].isnull().all():
                min_val = metrics_df[col].min()
                max_val = metrics_df[col].max()
                if max_val > min_val:
                    # For metrics where higher is better
                    if col in ['final_avg_reward', 'final_win_rate', 'reward_improvement', 'final_avg_q_value', 'final_max_q_value']:
                        metrics_df[f'{col}_norm'] = (metrics_df[col] - min_val) / (max_val - min_val)
                    # For metrics where lower is better
                    elif col in ['final_avg_steps', 'training_time']:
                        metrics_df[f'{col}_norm'] = 1 - (metrics_df[col] - min_val) / (max_val - min_val)
                    else:
                        metrics_df[f'{col}_norm'] = (metrics_df[col] - min_val) / (max_val - min_val)
                else:
                    metrics_df[f'{col}_norm'] = 0.5  # If all values are the same
        
        norm_cols = [f'{col}_norm' for col in radar_cols]
        
        # Set up the radar chart
        num_vars = len(norm_cols)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add the column names
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        # Plot each model
        for i, row in metrics_df.iterrows():
            values = [row[col] for col in norm_cols]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['model'])
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        labels = [col.replace('_norm', '').replace('_', ' ') for col in norm_cols]
        labels += labels[:1]  # Close the loop
        plt.xticks(angles, labels, fontsize=10)
        
        # Draw axis lines for each angle and add labels
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Model Training Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_radar_comparison.png'))
        plt.close()

    return metrics_df

def generate_comparison_report(all_data, individual_dfs, metrics_df):
    """
    Generate a markdown report summarizing the training comparison.
    """
    # Extract key insights
    model_names = list(individual_dfs.keys())
    fastest_training = metrics_df.loc[metrics_df['training_time'].idxmin()]['model'] if 'training_time' in metrics_df.columns and not metrics_df['training_time'].isnull().all() else 'Unknown'
    highest_final_reward = metrics_df.loc[metrics_df['final_avg_reward'].idxmax()]['model'] if 'final_avg_reward' in metrics_df.columns and not metrics_df['final_avg_reward'].isnull().all() else 'Unknown'
    highest_win_rate = metrics_df.loc[metrics_df['final_win_rate'].idxmax()]['model'] if 'final_win_rate' in metrics_df.columns and not metrics_df['final_win_rate'].isnull().all() else 'Unknown'
    
    # Generate the report
    report = f"""# Pacman RL Training Comparison Report

## Overview

This report compares the training performance of different Pacman RL models:
{', '.join([f'**{model}**' for model in model_names])}

## Key Findings

- **Highest Final Performance**: The {highest_final_reward} model achieved the highest final average reward.
- **Highest Win Rate**: The {highest_win_rate} model achieved the highest win rate.
- **Fastest Training**: The {fastest_training} model had the shortest training time.

## Training Metrics Summary

| Model | Episodes | Final Avg Reward | Final Win Rate | Avg Steps |
|-------|----------|----------------|---------------|----------|
"""
    
    # Add rows for each model
    for _, row in metrics_df.iterrows():
        final_reward = f"{row['final_avg_reward']:.2f}" if pd.notnull(row['final_avg_reward']) else "N/A"
        win_rate = f"{row['final_win_rate']:.2f}" if pd.notnull(row['final_win_rate']) else "N/A"
        avg_steps = f"{row['final_avg_steps']:.2f}" if pd.notnull(row['final_avg_steps']) else "N/A"
        
        report += f"| {row['model']} | {int(row['episodes'])} | {final_reward} | {win_rate} | {avg_steps} |\n"
    
    report += """
## Learning Curve Analysis

The learning curves show how each model's performance improved over the course of training.
Key observations:

"""
    
    # Add observations about learning curves
    # This could be enhanced with automated analysis of learning curve characteristics
    report += "- Learning rate and convergence patterns vary between models\n"
    report += "- See visualizations in the results directory for detailed comparisons\n"
    
    report += """
## Training Stability

Training stability measures how consistent the model's performance was during training.
Lower variance indicates more stable training.

"""
    
    # Add stability observations if available
    if 'total_reward' in all_data.columns:
        for model, df in individual_dfs.items():
            early_var = df['total_reward'].iloc[:int(len(df)*0.2)].var()
            late_var = df['total_reward'].iloc[int(len(df)*0.8):].var()
            report += f"- **{model}**: Early variance: {early_var:.2f}, Late variance: {late_var:.2f}\n"
    
    report += """
## Q-Value Analysis

Q-values represent the model's estimate of expected future rewards. Higher Q-values typically indicate a more confident model.

"""
    
    # Add Q-value observations if available
    if 'avg_q_value' in all_data.columns:
        for model, df in individual_dfs.items():
            final_avg_q = df['avg_q_value'].iloc[-100:].mean()
            report += f"- **{model}**: Final average Q-value: {final_avg_q:.2f}\n"
    
    report += """
## Win Rate Progression

Win rate shows how often the agent successfully completed the game over the course of training.

"""
    
    # Add win rate observations if available
    if 'win' in all_data.columns:
        for model, df in individual_dfs.items():
            early_win_rate = df['win'].iloc[:int(len(df)*0.2)].mean() * 100
            late_win_rate = df['win'].iloc[int(len(df)*0.8):].mean() * 100
            win_improvement = late_win_rate - early_win_rate
            report += f"- **{model}**: Early win rate: {early_win_rate:.2f}%, Late win rate: {late_win_rate:.2f}%, Improvement: {win_improvement:.2f}%\n"
    
    report += """
## Conclusion

Based on the analysis, the following conclusions can be drawn:

"""
    
    # Add conclusion based on performance
    report += f"1. **Best Overall Training Performance**: {highest_final_reward}\n"
    report += f"2. **Best Win Rate**: {highest_win_rate}\n"
    report += f"3. **Most Efficient Training**: {fastest_training}\n"
    report += "4. See the visualizations and metrics in the results directory for detailed comparisons\n"
    
    # Save the report
    report_path = os.path.join(RESULTS_DIR, 'training_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to {report_path}")

def analyze_training_logs(file_paths):
    """
    Analyze training logs and generate visualizations.
    
    Args:
        file_paths: Dictionary mapping model names to file paths
    """
    print("Loading training logs...")
    training_data = load_training_logs(file_paths)
    
    if not training_data:
        print("No training data could be loaded. Exiting.")
        return
    
    print("Normalizing episode count for fair comparison...")
    normalized_data = normalize_episode_count(training_data)
    
    print("Preprocessing training data...")
    all_data = preprocess_training_data(normalized_data)
    
    print("Generating learning curve plots...")
    generate_learning_curve_plots(all_data, normalized_data)
    
    print("Generating Q-value analysis...")
    generate_q_value_analysis(all_data, normalized_data)
    
    print("Generating exploration analysis...")
    generate_exploration_analysis(all_data, normalized_data)
    
    print("Generating performance distribution plots...")
    generate_performance_distribution_plots(all_data, normalized_data)
    
    print("Generating violin plots...")
    generate_violin_plots(all_data)
    
    print("Generating correlation heatmaps...")
    generate_correlation_heatmaps(all_data, normalized_data)
    
    print("Generating learning efficiency plots...")
    generate_learning_efficiency_plot(all_data, normalized_data)
    
    print("Generating milestone comparison...")
    generate_milestone_comparison(all_data, normalized_data)
    
    print("Generating win percentage over time...")
    generate_win_percentage_over_time(all_data, normalized_data)
    
    print("Generating training stability analysis...")
    generate_training_stability_analysis(all_data, normalized_data)
    
    print("Generating comparison metrics...")
    metrics_df = generate_comparison_metrics(all_data, normalized_data)
    
    print("Generating comparison report...")
    generate_comparison_report(all_data, normalized_data, metrics_df)
    
    print(f"All analyses and visualizations completed. Results are in the '{RESULTS_DIR}' directory.")

def main():
    """
    Main function to run the training log analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze Pacman RL training logs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing training log files')
    parser.add_argument('--fixed', type=str, default='logs/fixed.csv', help='Path to the fixed model training log')
    parser.add_argument('--progressive', type=str, default='logs/progressive.csv', help='Path to the progressive model training log')
    parser.add_argument('--domain_random', type=str, default='logs/domain_random.csv', help='Path to the domain randomization model training log')
    parser.add_argument('--random', type=str, default='logs/random.csv', help='Path to the random model training log')
    args = parser.parse_args()
    
    # Set up file paths
    file_paths = {
        'fixed': args.fixed,
        'progressive': args.progressive,
        'domain_random': args.domain_random,
        'random': args.random
    }
    
    # Analyze training logs
    analyze_training_logs(file_paths)

if __name__ == "__main__":
    main() 