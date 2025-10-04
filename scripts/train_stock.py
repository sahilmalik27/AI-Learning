#!/usr/bin/env python3
"""Easy script to train time series models for any stock ticker.

Usage:
    python scripts/train_stock.py --ticker AAPL --model lstm --epochs 20
    python scripts/train_stock.py --ticker TSLA --model gru --epochs 15
    python scripts/train_stock.py --ticker META --model rnn --epochs 10
"""

import sys
import os
import argparse
import yaml
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_config(ticker, model_type, epochs, hidden_size=128, window=120, horizon=20, data_source='yahoo'):
    """Create a temporary config file for the specified ticker and model."""
    config = {
        'run_name': f'{ticker.lower()}_{model_type}',
        'seed': 1337,
        'model': {
            'type': model_type,
            'hidden_size': hidden_size,
            'num_layers': 1,
            'bidirectional': False,
            'dropout': 0.0,
            'horizon': horizon
        },
        'data': {
            'type': data_source,
            'ticker': ticker,
            'use_log_returns': True,
            'window': window,
            'horizon': horizon,
            'split': 0.8
        },
        'train': {
            'epochs': epochs,
            'batch_size': 128,
            'lr': 0.001,
            'clip_grad': 1.0
        }
    }
    
    # Add data source specific parameters
    if data_source == 'yahoo':
        config['data']['period'] = '5y'
        config['data']['interval'] = '1d'
    elif data_source == 'stooq':
        config['data']['start_date'] = '2015-01-01'
        config['data']['end_date'] = '2025-01-01'
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Train time series models for any stock")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL, TSLA, META)")
    parser.add_argument("--model", default="lstm", choices=["rnn", "lstm", "gru"], help="Model type")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--window", type=int, default=120, help="Input window size (days of historical data)")
    parser.add_argument("--horizon", type=int, default=20, help="Prediction horizon (days to predict ahead)")
    parser.add_argument("--period", default="5y", help="Data period (1y, 2y, 5y, max)")
    parser.add_argument("--data-source", default="yahoo", choices=["yahoo", "stooq"], help="Data source (yahoo or stooq)")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date for Stooq (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-01-01", help="End date for Stooq (YYYY-MM-DD)")
    parser.add_argument("--predict-5-days", action="store_true", help="Set horizon=5 and window=60 for 5-day prediction")
    parser.add_argument("--no-fallback", action="store_true", help="Disable synthetic data fallback")
    
    args = parser.parse_args()
    
    # Handle 5-day prediction mode
    if args.predict_5_days:
        args.horizon = 5
        args.window = 60
        print("🎯 5-day prediction mode: horizon=5, window=60")
    
    # Create config
    config = create_config(
        ticker=args.ticker,
        model_type=args.model,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        window=args.window,
        horizon=args.horizon,
        data_source=args.data_source
    )
    
    # Update data source specific parameters
    if args.data_source == 'yahoo':
        config['data']['period'] = args.period
    elif args.data_source == 'stooq':
        config['data']['start_date'] = args.start_date
        config['data']['end_date'] = args.end_date
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name
    
    try:
        print(f"🚀 Training {args.model.upper()} model for {args.ticker}")
        print(f"📊 Config: {args.epochs} epochs, {args.hidden_size} hidden units, {args.window} window, {args.horizon} horizon")
        print(f"📈 Data source: {args.data_source.upper()}")
        if args.data_source == 'yahoo':
            print(f"📅 Data period: {args.period}")
        elif args.data_source == 'stooq':
            print(f"📅 Date range: {args.start_date} to {args.end_date}")
        
        # Import and run training
        from sequence_representations.exp_ts.train_ts import run_experiment
        run_experiment(config)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved to: models/representations/exp_ts/{config['run_name']}/best.pt")
        print(f"📊 Predictions saved to: predictions_ts/{config['run_name']}_val_preds.csv")
        
        # Show inference command
        print(f"\n🔮 To run inference:")
        print(f"python scripts/ts_infer.py --config {config_path} --ckpt models/representations/exp_ts/{config['run_name']}/best.pt")
        
    finally:
        # Clean up temporary file
        os.unlink(config_path)

if __name__ == "__main__":
    main()
