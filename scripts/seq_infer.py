#!/usr/bin/env python3
"""Inference script for sequence models.

Usage:
    python scripts/seq_infer.py --config src/sequence_representations/exp/configs/ag_news_lstm.yaml --ckpt models/representations/exp/lstm_ag_news/best.pt --text "Your news text here"
"""

import sys
import os
import argparse
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_representations.exp.data_text import tokenize
from sequence_representations.exp.models.sequence_models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for sequence models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--text", required=True, help="Text to classify")
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    
    # Rebuild vocab
    vocab = ckpt['vocab']  # This is already a dict
    
    # Build model
    model = build_model(cfg["model"], vocab_size=len(vocab))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    # Process text
    tokens = tokenize(args.text)
    ids = [vocab.get(tok, 0) for tok in tokens]  # 0 is <unk>
    ids = ids[:cfg["data"].get("max_len", 200)]  # Truncate if needed
    
    # Convert to tensor
    x = torch.tensor([ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    # AG_NEWS class names
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    print(f"Text: {args.text[:100]}...")
    print(f"Predicted class: {pred_class} ({class_names[pred_class]})")
    print(f"Confidence: {confidence:.3f}")
    print(f"All probabilities: {[f'{class_names[i]}: {probs[0, i]:.3f}' for i in range(4)]}")

if __name__ == "__main__":
    main()
