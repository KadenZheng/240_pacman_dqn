#!/usr/bin/env python3

"""
Minimal test script to verify the transition_mode fix works correctly.
"""

import os

# Set up directories
if not os.path.exists('saves'):
    os.makedirs('saves')

# Clean up any old models
for filename in os.listdir('saves'):
    if filename.startswith('model-'):
        os.remove(os.path.join('saves', filename))

# Run a single test with transition_mode=fixed
print("Testing fixed transition mode")
test_cmd = "python3 pacman.py -p PacmanDQN -n 3 -x 2 -l smallClassic -q --transition_mode=fixed"
print(f"Running: {test_cmd}")
os.system(test_cmd)

# Check if a model was saved
any_saved = False
for filename in os.listdir('saves'):
    if filename.startswith('model-fixed'):
        print(f"Found model: {filename}")
        any_saved = True

if any_saved:
    print("✅ Model saving works correctly")
else:
    print("❌ No models saved, fix did not work") 