"""Experiment runner for sequence models on AG_NEWS text classification.

Features:
- Model factory (RNN/LSTM/GRU)
- Device pick (CUDA→MPS→CPU)
- Config-driven training with AG_NEWS dataset
- TensorBoard logging with gate introspection
- Checkpointing to models/representations/exp/<exp_name>
"""

import argparse
import os
import yaml
from .engine import run_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence models on AG_NEWS")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs("models/representations/exp", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    
    run_experiment(cfg)

if __name__ == "__main__":
    main()
