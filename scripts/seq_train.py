#!/usr/bin/env python3
"""Launcher script for sequence model experiments.

Usage:
    python scripts/seq_train.py --config src/sequence_representations/exp/configs/ag_news_lstm.yaml
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_representations.exp.train import main

if __name__ == "__main__":
    main()
