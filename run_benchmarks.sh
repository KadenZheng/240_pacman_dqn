#!/bin/bash

# Number of games to run for each benchmark
GAMES=100

# Layout to use
LAYOUT="mediumClassic"

# Set up directories
mkdir -p logs/benchmarks
mkdir -p comparison_results
mkdir -p comparison_results/visualizations

echo "Starting model comparison benchmarking..."
python3 compare_all_models.py --games $GAMES --layout $LAYOUT

echo "Benchmarking complete. Results are in the 'comparison_results' directory."
echo "Visualizations and report are in the 'comparison_results' directory as well." 