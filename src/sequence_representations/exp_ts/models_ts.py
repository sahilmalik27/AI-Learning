"""Time series forecasting models: RNN, LSTM, GRU for stock price prediction.

Features:
- Univariate time series forecasting
- Multi-step ahead prediction
- Bidirectional support
- Configurable architecture
"""

import torch
import torch.nn as nn


class TSModel(nn.Module):
    """Time series forecasting model using RNN/LSTM/GRU."""
    
    def __init__(self, rnn_type='lstm', input_size=1, hidden_size=128, num_layers=1, 
                 bidirectional=False, dropout=0.0, horizon=16):
        super().__init__()
        RNN = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.horizon = horizon
        
        self.rnn = RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0, 
            bidirectional=bidirectional
        )
        
        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * mult, hidden_size * mult),
            nn.ReLU(),
            nn.Linear(hidden_size * mult, horizon)
        )

    def forward(self, x):
        """Forward pass for time series forecasting.
        
        Args:
            x: Input tensor of shape (batch, window, features)
            
        Returns:
            y: Predicted values of shape (batch, horizon)
        """
        H_all, H_last = self.rnn(x)
        
        if self.rnn_type == 'lstm':
            h_T = H_last[0]
        else:
            h_T = H_last
            
        if self.bidirectional:
            h_last_layer = h_T.view(self.rnn.num_layers, (2), h_T.size(1), h_T.size(2))[-1]
            h_cat = torch.cat([h_last_layer[0], h_last_layer[1]], dim=-1)
        else:
            h_cat = h_T[-1]
            
        y = self.head(h_cat)
        return y


def build_ts_model(cfg):
    """Build time series model from config."""
    return TSModel(
        rnn_type=cfg.get('type', 'lstm'),
        input_size=cfg.get('input_size', 1),
        hidden_size=cfg.get('hidden_size', 128),
        num_layers=cfg.get('num_layers', 1),
        bidirectional=cfg.get('bidirectional', False),
        dropout=cfg.get('dropout', 0.0),
        horizon=cfg.get('horizon', 16)
    )
