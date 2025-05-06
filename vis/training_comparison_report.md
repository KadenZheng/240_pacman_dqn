# Pacman RL Training Comparison Report

## Overview

This report compares the training performance of different Pacman RL models:
**fixed**, **progressive**, **domain_random**, **random**

## Key Findings

- **Highest Final Performance**: The progressive model achieved the highest final average reward.
- **Highest Win Rate**: The progressive model achieved the highest win rate.
- **Fastest Training**: The fixed model had the shortest training time.

## Training Metrics Summary

| Model | Episodes | Final Avg Reward | Final Win Rate | Avg Steps |
|-------|----------|----------------|---------------|----------|
| fixed | 5460 | 531.53 | 0.04 | 144.92 |
| progressive | 5460 | 679.26 | 0.29 | 169.47 |
| domain_random | 5460 | 574.39 | 0.17 | 160.26 |
| random | 5460 | 495.68 | 0.07 | 170.12 |

## Learning Curve Analysis

The learning curves show how each model's performance improved over the course of training.
Key observations:

- Learning rate and convergence patterns vary between models
- See visualizations in the results directory for detailed comparisons

## Training Stability

Training stability measures how consistent the model's performance was during training.
Lower variance indicates more stable training.

- **fixed**: Early variance: 5586.56, Late variance: 42390.98
- **progressive**: Early variance: 23326.53, Late variance: 71981.02
- **domain_random**: Early variance: 36152.03, Late variance: 85893.23
- **random**: Early variance: 13397.25, Late variance: 47240.36

## Q-Value Analysis

Q-values represent the model's estimate of expected future rewards. Higher Q-values typically indicate a more confident model.

- **fixed**: Final average Q-value: 91.72
- **progressive**: Final average Q-value: 98.67
- **domain_random**: Final average Q-value: 81.46
- **random**: Final average Q-value: 89.73

## Win Rate Progression

Win rate shows how often the agent successfully completed the game over the course of training.

- **fixed**: Early win rate: 0.00%, Late win rate: 2.38%, Improvement: 2.38%
- **progressive**: Early win rate: 0.00%, Late win rate: 18.41%, Improvement: 18.41%
- **domain_random**: Early win rate: 0.00%, Late win rate: 16.12%, Improvement: 16.12%
- **random**: Early win rate: 0.00%, Late win rate: 3.94%, Improvement: 3.94%

## Conclusion

Based on the analysis, the following conclusions can be drawn:

1. **Best Overall Training Performance**: progressive
2. **Best Win Rate**: progressive
3. **Most Efficient Training**: fixed
4. See the visualizations and metrics in the results directory for detailed comparisons
