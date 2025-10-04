#!/usr/bin/env python3
"""Inference script for time series forecasting models.

Usage:
    python scripts/ts_infer.py --config src/sequence_representations/exp_ts/configs/sine_lstm.yaml --ckpt models/representations/exp_ts/sine_lstm/best.pt --steps 5
"""

import sys
import os
import argparse
import torch
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_representations.exp_ts.models_ts import build_ts_model

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for time series models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=5, help="Number of future steps to predict")
    parser.add_argument("--input", type=str, help="Input sequence (comma-separated values)")
    parser.add_argument("--predict-5-days", action="store_true", help="Predict next 5 days (overrides --steps)")
    parser.add_argument("--show-dates", action="store_true", help="Show prediction dates")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle 5-day prediction mode
    if args.predict_5_days:
        args.steps = 5
        print("🎯 5-day prediction mode activated")
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Build model
    model = build_ts_model(cfg["model"])
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    # Prepare input
    if args.input:
        # Use provided input sequence
        input_values = [float(x.strip()) for x in args.input.split(',')]
        window_size = cfg["data"].get("window", 64)
        if len(input_values) < window_size:
            print(f"Warning: Input sequence length ({len(input_values)}) is less than window size ({window_size})")
            # Pad with zeros or repeat last value
            input_values = input_values + [input_values[-1]] * (window_size - len(input_values))
        elif len(input_values) > window_size:
            # Take the last window_size values
            input_values = input_values[-window_size:]
    else:
        # Generate synthetic input for demonstration
        window_size = cfg["data"].get("window", 64)
        t = np.linspace(0, 4*np.pi, window_size)
        input_values = np.sin(t).tolist()
        print(f"Using synthetic sine wave input (length: {len(input_values)})")
    
    # Convert to tensor
    x = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(x)
        pred_values = predictions[0].cpu().numpy()
    
    print(f"Input sequence (last {len(input_values)} values): {input_values[-10:]}")
    
    # Show predictions with dates if requested
    if args.show_dates:
        from datetime import datetime, timedelta
        today = datetime.now()
        print(f"\nPredictions for next {args.steps} days:")
        for i in range(min(args.steps, len(pred_values))):
            pred_date = today + timedelta(days=i+1)
            print(f"  {pred_date.strftime('%Y-%m-%d')} (Day {i+1}): {pred_values[i]:.4f}")
    else:
        print(f"\nPredictions for next {args.steps} days:")
        for i in range(min(args.steps, len(pred_values))):
            print(f"  Day {i+1}: {pred_values[i]:.4f}")
    
    # Show summary for 5-day prediction
    if args.predict_5_days:
        print(f"\n🎯 5-Day Prediction Summary:")
        print(f"  Based on last {len(input_values)} days of data")
        print(f"  Model trained to predict {len(pred_values)} days ahead")
        print(f"  Showing next 5 days:")
        for i in range(min(5, len(pred_values))):
            print(f"    Day {i+1}: {pred_values[i]:.4f}")
    
    print(f"\nModel: {cfg['model']['type'].upper()}")
    print(f"Window size: {len(input_values)}")
    print(f"Horizon: {len(pred_values)}")

if __name__ == "__main__":
    main()
