#!/usr/bin/env python3
"""Launcher script for time series forecasting experiments.

Usage:
    python scripts/ts_train.py --config src/sequence_representations/exp_ts/configs/yahoo_meta.yaml
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_representations.exp_ts.train_ts import main

if __name__ == "__main__":
    main()
