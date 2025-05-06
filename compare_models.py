#!/usr/bin/env python3
"""
compare_models.py
------------------
Script to compare the fixed and progressive models based on their logs.
Generates visualizations and metrics for comparison.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up directories
RESULTS_DIR = "comparison_results"
MODELS_DIR = "saves"

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_data():
    """Load the training data from CSV files"""
    fixed_data = pd.read_csv('logs/fixed.csv')
    progressive_data = pd.read_csv('logs/progressive.csv')
    
    # Add model type column for easier comparison
    fixed_data['model'] = 'Fixed'
    progressive_data['model'] = 'Progressive'
    
    return fixed_data, progressive_data

def compare_training_metrics(fixed_data, progressive_data):
    """Compare training metrics between models and save to CSV"""
    # Calculate key metrics
    metrics = {
        'Metric': [
            'Total Episodes', 
            'Max Reward', 
            'Final Avg Reward (last 100)',
            'Final Win Rate (last 100)', 
            'Final Avg Q-Value (last 100)',
            'Avg Steps per Episode',
            'Win Rate'
        ],
        'Fixed Model': [
            len(fixed_data),
            fixed_data['total_reward'].max(),
            fixed_data.iloc[-100:]['total_reward'].mean(),
            fixed_data.iloc[-100:]['win'].mean() * 100,
            fixed_data.iloc[-100:]['avg_q_value'].mean(),
            fixed_data['steps'].mean(),
            fixed_data['win'].mean() * 100
        ],
        'Progressive Model': [
            len(progressive_data),
            progressive_data['total_reward'].max(),
            progressive_data.iloc[-100:]['total_reward'].mean(),
            progressive_data.iloc[-100:]['win'].mean() * 100,
            progressive_data.iloc[-100:]['avg_q_value'].mean(),
            progressive_data['steps'].mean(),
            progressive_data['win'].mean() * 100
        ]
    }
    
    # Create comparison table
    metrics_df = pd.DataFrame(metrics)
    
    # Add difference column
    metrics_df['Difference'] = metrics_df['Progressive Model'] - metrics_df['Fixed Model']
    
    # Save to CSV
    metrics_df.to_csv(f"{RESULTS_DIR}/model_comparison_metrics.csv", index=False)
    
    return metrics_df

def generate_reward_comparison(fixed_data, progressive_data):
    """Generate reward comparison plot"""
    plt.figure(figsize=(12, 6))
    
    # Create a rolling window for smoother visualization
    window_size = 100
    fixed_rolling = fixed_data['total_reward'].rolling(window=window_size).mean()
    progressive_rolling = progressive_data['total_reward'].rolling(window=window_size).mean()
    
    # Plot both models
    plt.plot(fixed_data['episode'], fixed_rolling, label='Fixed Model', color='blue')
    plt.plot(progressive_data['episode'], progressive_rolling, label='Progressive Model', color='green')
    
    plt.title('Reward Comparison (100-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/reward_comparison.png")
    plt.close()

def generate_win_rate_comparison(fixed_data, progressive_data):
    """Generate win rate comparison plot"""
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative win rate
    fixed_data['cum_win_rate'] = fixed_data['win'].cumsum() / fixed_data['episode']
    progressive_data['cum_win_rate'] = progressive_data['win'].cumsum() / progressive_data['episode']
    
    # Create a rolling window for win rate
    window_size = 100
    fixed_win_rate = fixed_data['win'].rolling(window=window_size).mean()
    progressive_win_rate = progressive_data['win'].rolling(window=window_size).mean()
    
    # Plot both models
    plt.plot(fixed_data['episode'], fixed_win_rate, label='Fixed Model (Rolling)', color='blue')
    plt.plot(progressive_data['episode'], progressive_win_rate, label='Progressive Model (Rolling)', color='green')
    plt.plot(fixed_data['episode'], fixed_data['cum_win_rate'], label='Fixed Model (Cumulative)', color='blue', linestyle='--', alpha=0.5)
    plt.plot(progressive_data['episode'], progressive_data['cum_win_rate'], label='Progressive Model (Cumulative)', color='green', linestyle='--', alpha=0.5)
    
    plt.title('Win Rate Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/win_rate_comparison.png")
    plt.close()

def generate_q_value_comparison(fixed_data, progressive_data):
    """Generate Q-value comparison plot"""
    plt.figure(figsize=(12, 6))
    
    # Create a rolling window for smoother visualization
    window_size = 100
    fixed_rolling = fixed_data['avg_q_value'].rolling(window=window_size).mean()
    progressive_rolling = progressive_data['avg_q_value'].rolling(window=window_size).mean()
    
    # Plot both models
    plt.plot(fixed_data['episode'], fixed_rolling, label='Fixed Model', color='blue')
    plt.plot(progressive_data['episode'], progressive_rolling, label='Progressive Model', color='green')
    
    plt.title('Q-Value Comparison (100-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/q_value_comparison.png")
    plt.close()

def generate_training_stability(fixed_data, progressive_data):
    """Generate training stability comparison"""
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Define window size for rolling average
    window_size = 100
    
    # 1. Reward stability
    fixed_reward_std = fixed_data['total_reward'].rolling(window=window_size).std()
    progressive_reward_std = progressive_data['total_reward'].rolling(window=window_size).std()
    
    ax1.plot(fixed_data['episode'], fixed_reward_std, label='Fixed Model', color='blue')
    ax1.plot(progressive_data['episode'], progressive_reward_std, label='Progressive Model', color='green')
    ax1.set_title('Reward Stability (Standard Deviation over 100 episodes)')
    ax1.set_ylabel('Reward Std Dev')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Q-value stability
    fixed_q_std = fixed_data['avg_q_value'].rolling(window=window_size).std()
    progressive_q_std = progressive_data['avg_q_value'].rolling(window=window_size).std()
    
    ax2.plot(fixed_data['episode'], fixed_q_std, label='Fixed Model', color='blue')
    ax2.plot(progressive_data['episode'], progressive_q_std, label='Progressive Model', color='green')
    ax2.set_title('Q-Value Stability (Standard Deviation over 100 episodes)')
    ax2.set_ylabel('Q-Value Std Dev')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Steps stability
    fixed_steps_std = fixed_data['steps'].rolling(window=window_size).std()
    progressive_steps_std = progressive_data['steps'].rolling(window=window_size).std()
    
    ax3.plot(fixed_data['episode'], fixed_steps_std, label='Fixed Model', color='blue')
    ax3.plot(progressive_data['episode'], progressive_steps_std, label='Progressive Model', color='green')
    ax3.set_title('Steps Stability (Standard Deviation over 100 episodes)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps Std Dev')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_stability_comparison.png")
    plt.close()

def generate_scatter_comparison(fixed_data, progressive_data):
    """Generate scatter plot to compare rewards vs steps for both models"""
    plt.figure(figsize=(10, 8))
    
    # Plot scatter points
    plt.scatter(fixed_data['steps'], fixed_data['total_reward'], 
               alpha=0.5, label='Fixed Model', color='blue')
    plt.scatter(progressive_data['steps'], progressive_data['total_reward'], 
               alpha=0.5, label='Progressive Model', color='green')
    
    # Add trend lines
    fixed_z = np.polyfit(fixed_data['steps'], fixed_data['total_reward'], 1)
    progressive_z = np.polyfit(progressive_data['steps'], progressive_data['total_reward'], 1)
    
    fixed_p = np.poly1d(fixed_z)
    progressive_p = np.poly1d(progressive_z)
    
    plt.plot(sorted(fixed_data['steps']), fixed_p(sorted(fixed_data['steps'])), 
            color='blue', linestyle='--')
    plt.plot(sorted(progressive_data['steps']), progressive_p(sorted(progressive_data['steps'])), 
            color='green', linestyle='--')
    
    plt.title('Reward vs. Steps Correlation')
    plt.xlabel('Steps per Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/reward_steps_correlation.png")
    plt.close()

def generate_training_efficiency(fixed_data, progressive_data):
    """Generate training efficiency comparison"""
    # Calculate cumulative rewards
    fixed_data['cum_reward'] = fixed_data['total_reward'].cumsum()
    progressive_data['cum_reward'] = progressive_data['total_reward'].cumsum()
    
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative rewards
    plt.plot(fixed_data['episode'], fixed_data['cum_reward'], label='Fixed Model', color='blue')
    plt.plot(progressive_data['episode'], progressive_data['cum_reward'], label='Progressive Model', color='green')
    
    plt.title('Cumulative Reward (Training Efficiency)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_efficiency.png")
    plt.close()

def generate_histogram_comparison(fixed_data, progressive_data):
    """Generate histogram comparisons for various metrics"""
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Reward distribution (last 1000 episodes)
    sns.histplot(data=fixed_data.iloc[-1000:], x='total_reward', kde=True, color='blue', alpha=0.6, ax=ax1, label='Fixed')
    sns.histplot(data=progressive_data.iloc[-1000:], x='total_reward', kde=True, color='green', alpha=0.6, ax=ax1, label='Progressive')
    ax1.set_title('Reward Distribution (Last 1000 Episodes)')
    ax1.set_xlabel('Total Reward')
    ax1.legend()
    
    # 2. Q-value distribution (last 1000 episodes)
    sns.histplot(data=fixed_data.iloc[-1000:], x='avg_q_value', kde=True, color='blue', alpha=0.6, ax=ax2, label='Fixed')
    sns.histplot(data=progressive_data.iloc[-1000:], x='avg_q_value', kde=True, color='green', alpha=0.6, ax=ax2, label='Progressive')
    ax2.set_title('Q-Value Distribution (Last 1000 Episodes)')
    ax2.set_xlabel('Average Q-Value')
    ax2.legend()
    
    # 3. Steps distribution
    sns.histplot(data=fixed_data, x='steps', kde=True, color='blue', alpha=0.6, ax=ax3, label='Fixed')
    sns.histplot(data=progressive_data, x='steps', kde=True, color='green', alpha=0.6, ax=ax3, label='Progressive')
    ax3.set_title('Steps Distribution')
    ax3.set_xlabel('Steps per Episode')
    ax3.legend()
    
    # 4. Win distribution (comparing 0 and 1 values)
    # Create a dataframe for win counts
    win_counts = pd.DataFrame({
        'Model': ['Fixed', 'Progressive'],
        'Win': [fixed_data['win'].sum(), progressive_data['win'].sum()],
        'Lose': [len(fixed_data) - fixed_data['win'].sum(), len(progressive_data) - progressive_data['win'].sum()]
    })
    
    # Melt the dataframe for easier plotting
    win_counts_melted = pd.melt(win_counts, id_vars=['Model'], var_name='Outcome', value_name='Count')
    
    # Create a grouped bar plot
    sns.barplot(x='Model', y='Count', hue='Outcome', data=win_counts_melted, ax=ax4)
    ax4.set_title('Win/Loss Distribution')
    ax4.set_ylabel('Count')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metric_distributions.png")
    plt.close()

def generate_comparison_report(metrics_df, fixed_data, progressive_data):
    """Generate a markdown report comparing the models"""
    report_path = os.path.join(RESULTS_DIR, "model_comparison_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Pacman DQN Model Comparison Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Model Overview\n\n")
        f.write("This report compares two Pacman DQN training approaches:\n\n")
        f.write("1. **Fixed Model**: Uses a fixed ghost behavior throughout training\n")
        f.write("2. **Progressive Model**: Uses a progressive training approach\n\n")
        
        f.write("## Key Performance Metrics\n\n")
        
        # Format the metrics as a markdown table
        f.write("| Metric | Fixed Model | Progressive Model | Difference |\n")
        f.write("|--------|-------------|------------------|------------|\n")
        
        for _, row in metrics_df.iterrows():
            metric = row['Metric']
            fixed_val = row['Fixed Model']
            prog_val = row['Progressive Model']
            diff = row['Difference']
            
            # Format the values based on the metric type
            if metric in ['Total Episodes']:
                fixed_val = f"{int(fixed_val)}"
                prog_val = f"{int(prog_val)}"
                diff = f"{int(diff)}"
            elif metric in ['Final Win Rate (last 100)', 'Win Rate']:
                fixed_val = f"{fixed_val:.2f}%"
                prog_val = f"{prog_val:.2f}%"
                diff = f"{diff:.2f}%"
            else:
                fixed_val = f"{fixed_val:.2f}"
                prog_val = f"{prog_val:.2f}"
                diff = f"{diff:.2f}"
                
            f.write(f"| {metric} | {fixed_val} | {prog_val} | {diff} |\n")
        
        f.write("\n## Training Progression\n\n")
        
        # Add information about the training progression
        fixed_episodes = len(fixed_data)
        prog_episodes = len(progressive_data)
        
        f.write(f"The fixed model was trained for {fixed_episodes} episodes, ")
        f.write(f"while the progressive model was trained for {prog_episodes} episodes.\n\n")
        
        f.write("### Reward Progression\n\n")
        f.write("![Reward Comparison](reward_comparison.png)\n\n")
        
        f.write("### Win Rate Progression\n\n")
        f.write("![Win Rate Comparison](win_rate_comparison.png)\n\n")
        
        f.write("### Q-Value Progression\n\n")
        f.write("![Q-Value Comparison](q_value_comparison.png)\n\n")
        
        f.write("## Training Stability\n\n")
        f.write("![Training Stability](training_stability_comparison.png)\n\n")
        
        f.write("## Training Efficiency\n\n")
        f.write("![Training Efficiency](training_efficiency.png)\n\n")
        
        f.write("## Metric Distributions\n\n")
        f.write("![Metric Distributions](metric_distributions.png)\n\n")
        
        f.write("## Reward vs. Steps Correlation\n\n")
        f.write("![Reward vs Steps](reward_steps_correlation.png)\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Determine which model performed better based on the metrics
        if metrics_df.loc[metrics_df['Metric'] == 'Final Win Rate (last 100)', 'Difference'].values[0] > 0:
            better_model = "Progressive"
            worse_model = "Fixed"
        else:
            better_model = "Fixed"
            worse_model = "Progressive"
        
        win_rate_diff = abs(metrics_df.loc[metrics_df['Metric'] == 'Final Win Rate (last 100)', 'Difference'].values[0])
        reward_diff = abs(metrics_df.loc[metrics_df['Metric'] == 'Final Avg Reward (last 100)', 'Difference'].values[0])
        
        f.write(f"Based on the analysis, the **{better_model} Model** demonstrates better overall performance ")
        f.write(f"with a {win_rate_diff:.2f}% higher win rate in the final stages of training compared to the {worse_model} Model. ")
        
        # Add final observations based on the data
        if better_model == "Progressive":
            f.write("The Progressive Model's approach of adapting to different ghost behaviors during training appears to be more effective ")
            f.write("at developing robust strategies that can generalize to different game scenarios.\n\n")
        else:
            f.write("The Fixed Model's approach of consistent ghost behavior during training appears to be more effective ")
            f.write("at developing specialized strategies for known game scenarios.\n\n")
        
    print(f"Comparison report saved to {report_path}")

def main():
    """Main function to run the comparison script"""
    # Set the style for plots
    sns.set(style="whitegrid")
    plt.style.use('ggplot')
    
    print("Loading training data...")
    fixed_data, progressive_data = load_data()
    
    print("Comparing training metrics...")
    metrics_df = compare_training_metrics(fixed_data, progressive_data)
    
    print("Generating visualizations...")
    generate_reward_comparison(fixed_data, progressive_data)
    generate_win_rate_comparison(fixed_data, progressive_data)
    generate_q_value_comparison(fixed_data, progressive_data)
    generate_training_stability(fixed_data, progressive_data)
    generate_scatter_comparison(fixed_data, progressive_data)
    generate_training_efficiency(fixed_data, progressive_data)
    generate_histogram_comparison(fixed_data, progressive_data)
    
    print("Generating comparison report...")
    generate_comparison_report(metrics_df, fixed_data, progressive_data)
    
    print(f"Model comparison complete. Results saved to {RESULTS_DIR} directory.")

if __name__ == "__main__":
    main()