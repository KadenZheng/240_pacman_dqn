import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import glob

def load_all_benchmark_files(dir_path='logs/benchmarks_v'):
    """Load all CSV benchmark files and return them as a dictionary with timestamps as keys."""
    benchmarks = {}
    
    # Define mapping for transition modes to readable labels
    mode_labels = {
        'fixed': 'Fixed Ghost Behavior',
        'progressive': 'Progressive Shifts',
        'domain_random': 'Domain Randomization',
        'random': 'Random Shifts',
    }
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
    
    # Sort log files by timestamp for potential matching by order
    sorted_logs = sorted(csv_files, key=lambda x: os.path.basename(x).split('-')[0])
    expected_order = ['fixed', 'progressive', 'domain_random', 'random']
    
    file_labels = {}
    
    # First, try to determine modes from file content
    for f in csv_files:
        timestamp = os.path.basename(f).split('-')[0] + '-' + os.path.basename(f).split('-')[1]
        
        # Try to read the file to find transition mode references
        transition_mode = None
        try:
            with open(f, 'r') as file:
                content = file.read(10000)  # Read first 10KB
                for mode in mode_labels.keys():
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
        except:
            pass
    
    # For files without determined mode, use order-based inference
    if len(file_labels) < len(csv_files):
        for i, f in enumerate(sorted_logs):
            if f not in file_labels and i < len(expected_order):
                file_labels[f] = mode_labels[expected_order[i]]
    
    # Load data and use labels
    for file_path in csv_files:
        # Extract timestamp from filename (still needed as dictionary key)
        filename = os.path.basename(file_path)
        timestamp = filename.split('-')[0] + '-' + filename.split('-')[1]
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Add a label column based on the file's transition mode
        label = file_labels.get(file_path, timestamp)  # Default to timestamp if no label found
        df['label'] = label
        
        # Add to dictionary
        benchmarks[timestamp] = df
    
    return benchmarks, file_labels

def smooth_data(data, window=50):
    """Apply a moving average smoothing to the data."""
    return data.rolling(window=window, min_periods=1).mean()

def analyze_training_progression(benchmark_data, file_labels):
    """Analyze training progression over episodes."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training Progression Analysis', fontsize=16)
    
    # Plot smoothed reward curves for all benchmarks
    ax = axs[0, 0]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        ax.plot(df['episode'], smooth_data(df['total_reward']), label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (Smoothed)')
    ax.set_title('Reward Progression')
    ax.grid(True)
    
    # Plot Q-values
    ax = axs[0, 1]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        ax.plot(df['episode'], smooth_data(df['avg_q_value']), label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Q-Value (Smoothed)')
    ax.set_title('Q-Value Progression')
    ax.grid(True)
    
    # Plot win rate
    ax = axs[1, 0]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        # Calculate win rate with a moving window
        win_rate = df['win'].rolling(window=100, min_periods=1).mean()
        ax.plot(df['episode'], win_rate, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (100-episode window)')
    ax.set_title('Win Rate Progression')
    ax.grid(True)
    
    # Plot steps per episode
    ax = axs[1, 1]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        ax.plot(df['episode'], smooth_data(df['steps']), label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode (Smoothed)')
    ax.set_title('Survival Time Progression')
    ax.grid(True)
    
    # Plot epsilon decay
    ax = axs[2, 0]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        ax.plot(df['episode'], df['epsilon'], label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon Value')
    ax.set_title('Exploration Rate (Epsilon) Decay')
    ax.grid(True)
    
    # Plot Q-value min/max range
    ax = axs[2, 1]
    for file_path, df in benchmark_data.items():
        label = df['label'].iloc[0]  # Get the label from the dataframe
        # We'll just use one benchmark for this to avoid cluttering
        q_range = df['max_q_value'] - df['min_q_value']
        ax.plot(df['episode'], smooth_data(q_range), label=f"{label} (range)")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-Value Range (Smoothed)')
    ax.set_title('Q-Value Range (Max - Min)')
    ax.grid(True)
    
    # Add a legend to the first subplot and adjust layout
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.98))
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    return fig

def compute_performance_metrics(benchmark_data):
    """Compute performance metrics for each benchmark."""
    metrics = {}
    
    for timestamp, df in benchmark_data.items():
        # Skip initial exploration phase (first 1000 episodes or until epsilon < 0.5)
        training_start_idx = df[df['epsilon'] < 0.5].index[0] if any(df['epsilon'] < 0.5) else 1000
        training_df = df.iloc[training_start_idx:]
        
        # Last 1000 episodes for final performance
        final_df = df.iloc[-1000:] if len(df) > 1000 else df
        
        metrics[timestamp] = {
            'avg_reward': final_df['total_reward'].mean(),
            'max_reward': final_df['total_reward'].max(),
            'avg_q_value': final_df['avg_q_value'].mean(),
            'final_win_rate': final_df['win'].mean() * 100,  # as percentage
            'avg_survival_steps': final_df['steps'].mean(),
            'learning_stability': np.abs(np.diff(smooth_data(training_df['avg_q_value']).dropna())).mean(),
            'reward_variance': final_df['total_reward'].var(),
            'q_value_variance': final_df['avg_q_value'].var(),
        }
    
    return pd.DataFrame(metrics).T

def analyze_q_value_distribution(benchmark_data):
    """Analyze Q-value distribution and stability."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Q-Value Distribution Analysis', fontsize=16)
    
    # For each benchmark, analyze Q-value distributions at different training stages
    for i, (timestamp, df) in enumerate(benchmark_data.items()):
        if i >= 4:  # Limit to 4 benchmarks for clarity
            continue
            
        # Define training stages (early, middle, late)
        early = df.iloc[:len(df)//3]
        middle = df.iloc[len(df)//3:2*len(df)//3]
        late = df.iloc[2*len(df)//3:]
        
        ax = axs[i // 2, i % 2]
        
        # Get the label from the dataframe instead of using timestamp
        label = df['label'].iloc[0]
        
        # Plot Q-value distributions for different stages
        sns.kdeplot(early['avg_q_value'].dropna(), ax=ax, label='Early Training')
        sns.kdeplot(middle['avg_q_value'].dropna(), ax=ax, label='Middle Training')
        sns.kdeplot(late['avg_q_value'].dropna(), ax=ax, label='Late Training')
        
        ax.set_xlabel('Average Q-Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Q-Value Distribution for {label}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def analyze_reward_stability(benchmark_data):
    """Analyze reward stability and convergence."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reward Stability Analysis', fontsize=16)
    
    # For each benchmark, analyze reward stability
    for i, (timestamp, df) in enumerate(benchmark_data.items()):
        if i >= 4:  # Limit to 4 benchmarks for clarity
            continue
            
        # Calculate moving statistics
        window_size = 100
        reward_mean = df['total_reward'].rolling(window=window_size, min_periods=1).mean()
        reward_std = df['total_reward'].rolling(window=window_size, min_periods=1).std()
        
        ax = axs[i // 2, i % 2]
        
        # Get the label from the dataframe instead of using timestamp
        label = df['label'].iloc[0]
        
        # Plot reward mean and standard deviation
        ax.plot(df['episode'], reward_mean, label='Mean Reward (100-episode window)')
        ax.fill_between(df['episode'], 
                        reward_mean - reward_std, 
                        reward_mean + reward_std, 
                        alpha=0.3, 
                        label='±1 Standard Deviation')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'Reward Stability for {label}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def correlation_analysis(benchmark_data):
    """Analyze correlations between metrics."""
    # Combine all benchmark data
    combined_df = pd.concat([df.assign(benchmark=timestamp) for timestamp, df in benchmark_data.items()])
    
    # Calculate correlation matrix
    correlation_metrics = ['total_reward', 'avg_q_value', 'steps', 'epsilon', 'win']
    correlation_matrix = combined_df[correlation_metrics].corr()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Between Training Metrics')
    
    return fig, correlation_matrix

def compare_benchmarks_summary(benchmark_data):
    """Generate a summary comparison of benchmarks."""
    summary = {}
    
    for timestamp, df in benchmark_data.items():
        label = df['label'].iloc[0]
        # Calculate overall statistics
        summary[label] = {  # Use label instead of timestamp
            'episodes': len(df),
            'final_epsilon': df['epsilon'].iloc[-1],
            'max_q_value_achieved': df['max_q_value'].max(),
            'avg_reward_last_1000': df['total_reward'].iloc[-1000:].mean() if len(df) > 1000 else df['total_reward'].mean(),
            'win_rate_last_1000': df['win'].iloc[-1000:].mean() * 100 if len(df) > 1000 else df['win'].mean() * 100,
            'avg_steps_last_1000': df['steps'].iloc[-1000:].mean() if len(df) > 1000 else df['steps'].mean(),
            'reward_stability': df['total_reward'].iloc[-1000:].std() if len(df) > 1000 else df['total_reward'].std(),
        }
    
    return pd.DataFrame(summary).T

def generate_full_report(benchmarks_dir='logs/benchmarks_v', output_dir='analysis_results'):
    """Generate a full analysis report with visualizations and metrics."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load benchmark data
    benchmark_data, file_labels = load_all_benchmark_files(benchmarks_dir)
    
    # Generate and save training progression plots
    fig_progression = analyze_training_progression(benchmark_data, file_labels)
    fig_progression.savefig(os.path.join(output_dir, 'training_progression.png'))
    
    # Generate and save Q-value distribution analysis
    fig_q_distribution = analyze_q_value_distribution(benchmark_data)
    fig_q_distribution.savefig(os.path.join(output_dir, 'q_value_distribution.png'))
    
    # Generate and save reward stability analysis
    fig_reward_stability = analyze_reward_stability(benchmark_data)
    fig_reward_stability.savefig(os.path.join(output_dir, 'reward_stability.png'))
    
    # Generate and save correlation analysis
    fig_correlation, correlation_matrix = correlation_analysis(benchmark_data)
    fig_correlation.savefig(os.path.join(output_dir, 'correlation_analysis.png'))
    
    # Compute performance metrics
    performance_metrics = compute_performance_metrics(benchmark_data)
    performance_metrics.to_csv(os.path.join(output_dir, 'performance_metrics.csv'))
    
    # Generate benchmark summary
    benchmark_summary = compare_benchmarks_summary(benchmark_data)
    benchmark_summary.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'))
    
    # Generate a text report summarizing findings
    generate_text_report(benchmark_data, performance_metrics, correlation_matrix, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    return benchmark_data, performance_metrics

def generate_text_report(benchmark_data, performance_metrics, correlation_matrix, output_dir):
    """Generate a text report summarizing the analysis findings."""
    report = []
    report.append("# PacMan DQN Training Analysis Report")
    report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Benchmark Overview")
    for timestamp, df in benchmark_data.items():
        label = df['label'].iloc[0]
        report.append(f"- **{label}**: {len(df)} episodes, final ε={df['epsilon'].iloc[-1]:.4f}")
    report.append("")
    
    report.append("## Performance Metrics Summary")
    report.append("```")
    # Create a copy of performance metrics with readable index labels
    readable_metrics = performance_metrics.copy()
    label_mapping = {timestamp: benchmark_data[timestamp]['label'].iloc[0] for timestamp in benchmark_data.keys()}
    readable_metrics.index = [label_mapping.get(idx, idx) for idx in readable_metrics.index]
    report.append(readable_metrics.to_string())
    report.append("```\n")
    
    report.append("## Key Observations")
    
    # Add observations about reward trends
    reward_trends = []
    for timestamp, df in benchmark_data.items():
        label = df['label'].iloc[0]
        early_rewards = df['total_reward'].iloc[:1000].mean() if len(df) > 1000 else df['total_reward'].iloc[:len(df)//3].mean()
        late_rewards = df['total_reward'].iloc[-1000:].mean() if len(df) > 1000 else df['total_reward'].iloc[2*len(df)//3:].mean()
        improvement = late_rewards - early_rewards
        reward_trends.append((label, improvement))
    
    best_improvement = max(reward_trends, key=lambda x: x[1])
    report.append(f"- The benchmark with the most reward improvement is **{best_improvement[0]}** with an increase of {best_improvement[1]:.2f} points.")
    
    # Add observations about win rate
    win_rates = [(benchmark_data[timestamp]['label'].iloc[0], df['win'].iloc[-1000:].mean() if len(df) > 1000 else df['win'].mean()) for timestamp, df in benchmark_data.items()]
    best_win_rate = max(win_rates, key=lambda x: x[1])
    report.append(f"- The benchmark with the highest win rate is **{best_win_rate[0]}** at {best_win_rate[1]*100:.2f}%.")
    
    # Add observations about Q-value stability
    q_stability = [(benchmark_data[timestamp]['label'].iloc[0], df['avg_q_value'].iloc[-1000:].std() if len(df) > 1000 else df['avg_q_value'].dropna().std()) for timestamp, df in benchmark_data.items()]
    most_stable = min(q_stability, key=lambda x: x[1])
    report.append(f"- The benchmark with the most stable Q-values is **{most_stable[0]}** with a standard deviation of {most_stable[1]:.4f}.")
    
    # Add observations about correlation
    strongest_corr = None
    strongest_val = 0
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > abs(strongest_val):
                strongest_val = correlation_matrix.iloc[i, j]
                strongest_corr = (correlation_matrix.columns[i], correlation_matrix.columns[j])
    
    if strongest_corr:
        report.append(f"- The strongest correlation ({strongest_val:.2f}) is between **{strongest_corr[0]}** and **{strongest_corr[1]}**.")
    
    report.append("\n## Recommendations")
    report.append("Based on the analysis, the following recommendations can be made:")
    
    # Add specific recommendations based on findings
    best_performing_idx = performance_metrics['avg_reward'].idxmax()
    best_performing_label = benchmark_data[best_performing_idx]['label'].iloc[0]
    report.append(f"1. The configuration from **{best_performing_label}** shows the best overall performance.")
    
    # Compare exploration strategies
    report.append("2. Exploration strategy comparison:")
    for timestamp, df in benchmark_data.items():
        label = df['label'].iloc[0]
        eps_half_point = df[df['epsilon'] <= 0.5].iloc[0]['episode'] if any(df['epsilon'] <= 0.5) else "N/A"
        report.append(f"   - **{label}**: Reached ε=0.5 at episode {eps_half_point}")
    
    # Save the report
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.write('\n'.join(report))

if __name__ == "__main__":
    generate_full_report()