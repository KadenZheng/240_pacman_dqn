#!/bin/bash

# Number of training/test episodes
TRAIN=5000
TEST=1000
TOTAL=$((TRAIN + TEST))

# Layout to use
LAYOUT="mediumClassic"

# Set up directories
mkdir -p logs
mkdir -p saves

# Clean up any existing models before starting
echo "Cleaning up old models..."
rm -f saves/model-*

echo "Running training with different ghost transition modes..."
echo "Note: Only final models will be saved at the end of training."

echo "Running baseline training with fixed transition function..."
python3 pacman.py -p PacmanDQN -n $TOTAL -x $TRAIN -l $LAYOUT -q --transition_mode=fixed

echo "Running training with progressive transition function shifts..."
python3 pacman.py -p PacmanDQN -n $TOTAL -x $TRAIN -l $LAYOUT -q --transition_mode=progressive

echo "Running training with domain randomization..."
python3 pacman.py -p PacmanDQN -n $TOTAL -x $TRAIN -l $LAYOUT -q --transition_mode=domain_random

echo "Running training with random transition function shifts..."
python3 pacman.py -p PacmanDQN -n $TOTAL -x $TRAIN -l $LAYOUT -q --transition_mode=random

echo "Training complete. Final models are saved in the 'saves' directory:"
ls -l saves/

echo "Running visualizations..."
python3 plot_metrics.py

echo "Run './run_benchmarks.sh' to test your trained models against different ghost behaviors." 