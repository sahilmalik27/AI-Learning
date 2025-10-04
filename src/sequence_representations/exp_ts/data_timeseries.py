"""Time series data utilities with Yahoo Finance support.

Features:
- Synthetic sine wave generation
- CSV file loading
- Yahoo Finance data download (AAPL, META, BTC-USD, etc.)
- Sliding window dataset creation
- Normalization and train/test splitting
"""

import torch
from torch.utils.data import Dataset
import math
import random
import pandas as pd
import numpy as np


class WindowDS(Dataset):
    """Sliding window dataset for time series forecasting."""
    
    def __init__(self, series, window, horizon, train=True, split=0.8, norm_by_train=True):
        N = len(series)
        cutoff = int(N*split)
        self.window = window
        self.horizon = horizon
        
        if train:
            s = series[:cutoff]
            if norm_by_train:
                self.mean = s.mean()
                self.std = s.std() + 1e-8
                s = (s - self.mean) / self.std
        else:
            s = series[cutoff-window-horizon:]
            if norm_by_train and hasattr(self, 'mean'):
                s = (s - self.mean) / self.std
                
        self.X, self.Y = self.build_xy(s, window, horizon)

    def build_xy(self, s, window, horizon):
        """Build input-output pairs from time series."""
        X, Y = [], []
        for i in range(len(s) - window - horizon + 1):
            X.append(s[i:i+window])
            Y.append(s[i+window:i+window+horizon])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_sine_series(N=5000, freq=0.01, trend=0.0005, noise=0.1, seed=123):
    """Generate synthetic sine wave with trend and noise."""
    random.seed(seed)
    t = np.arange(N)
    s = np.sin(2*math.pi*freq*t) + trend*t + np.random.normal(0, noise, size=N)
    return s.astype('float32')


def load_csv_series(path, value_col='value'):
    """Load time series from CSV file."""
    df = pd.read_csv(path)
    s = df[value_col].astype('float32').values
    return s


def generate_synthetic_stock_data(ticker='AAPL', days=1000, seed=42):
    """Generate synthetic stock-like data for testing when Yahoo Finance is unavailable."""
    np.random.seed(seed)
    
    # Generate realistic stock price data
    # Start with a base price
    base_price = 100.0
    
    # Generate daily returns with some autocorrelation
    returns = np.random.normal(0.0005, 0.02, days)  # Small positive drift, 2% daily volatility
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Some momentum
    
    # Convert returns to prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove the initial base price
    
    print(f"Generated synthetic {ticker} data: {len(prices)} days")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    return prices.astype('float32')


def load_stooq_series(ticker='AAPL', start_date='2015-01-01', end_date='2025-01-01', use_log_returns=True, fallback_to_synthetic=True):
    """Load stock data from Stooq using pandas-datareader.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'META', 'TSLA', 'GOOGL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_log_returns: If True, return log returns; if False, return raw prices
        fallback_to_synthetic: If True, generate synthetic data when Stooq fails
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        raise ImportError("pandas-datareader is required for Stooq data. Install with: pip install pandas-datareader")
    
    print(f"Downloading {ticker} data from Stooq...")
    
    try:
        # Use pandas-datareader to get data from Stooq
        df = web.DataReader(ticker, "stooq", start=start_date, end=end_date).sort_index()
        
        if df.empty or len(df) == 0:
            raise ValueError(f"No data found for ticker {ticker}")
        
        close = df['Close'].astype('float32').values
        print(f"✅ Downloaded {len(close)} data points for {ticker} from Stooq")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Stooq failed: {e}")
        if fallback_to_synthetic:
            print(f"🔄 Generating synthetic {ticker} data as fallback...")
            close = generate_synthetic_stock_data(ticker, days=1000)
        else:
            raise ValueError(f"Failed to download {ticker} data and fallback disabled")
    
    if use_log_returns:
        close = close[~pd.isna(close)]
        if len(close) < 2:
            raise ValueError(f"Not enough data points for log returns calculation. Got {len(close)} points.")
        r = np.diff(np.log(close))  # log returns
        print(f"Computed {len(r)} log returns")
        return r.astype('float32')
    else:
        return close


def load_yahoo_series(ticker='AAPL', period='5y', interval='1d', use_log_returns=True, fallback_to_synthetic=True):
    """Load stock data from Yahoo Finance using Ticker object.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'META', 'BTC-USD', 'TSLA', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        use_log_returns: If True, return log returns; if False, return raw prices
        fallback_to_synthetic: If True, generate synthetic data when Yahoo Finance fails
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for Yahoo Finance data. Install with: pip install yfinance")
    
    print(f"Downloading {ticker} data from Yahoo Finance...")
    
    try:
        # Use Ticker object for more reliable data access
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty or len(df) == 0:
            raise ValueError(f"No data found for ticker {ticker}")
        
        close = df['Close'].astype('float32').values
        print(f"✅ Downloaded {len(close)} data points for {ticker}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Yahoo Finance failed: {e}")
        if fallback_to_synthetic:
            print(f"🔄 Generating synthetic {ticker} data as fallback...")
            close = generate_synthetic_stock_data(ticker, days=1000)
        else:
            raise ValueError(f"Failed to download {ticker} data and fallback disabled")
    
    if use_log_returns:
        close = close[~pd.isna(close)]
        if len(close) < 2:
            raise ValueError(f"Not enough data points for log returns calculation. Got {len(close)} points.")
        r = np.diff(np.log(close))  # log returns
        print(f"Computed {len(r)} log returns")
        return r.astype('float32')
    else:
        return close


def build_datasets(cfg):
    """Build train/val datasets from various sources."""
    src = cfg.get('type', 'sine')
    
    if src == 'sine':
        s = make_sine_series(
            N=cfg.get('N', 5000), 
            freq=cfg.get('freq', 0.01), 
            trend=cfg.get('trend', 0.0), 
            noise=cfg.get('noise', 0.1)
        )
    elif src == 'csv':
        s = load_csv_series(cfg['path'], cfg.get('value_col', 'value'))
    elif src == 'yahoo':
        s = load_yahoo_series(
            ticker=cfg.get('ticker', 'BTC-USD'),
            period=cfg.get('period', '5y'),
            interval=cfg.get('interval', '1d'),
            use_log_returns=cfg.get('use_log_returns', True),
            fallback_to_synthetic=cfg.get('fallback_to_synthetic', True)
        )
    elif src == 'stooq':
        s = load_stooq_series(
            ticker=cfg.get('ticker', 'AAPL'),
            start_date=cfg.get('start_date', '2015-01-01'),
            end_date=cfg.get('end_date', '2025-01-01'),
            use_log_returns=cfg.get('use_log_returns', True),
            fallback_to_synthetic=cfg.get('fallback_to_synthetic', True)
        )
    else:
        raise ValueError(f"Unknown data source: {src}")

    train_ds = WindowDS(
        s, 
        cfg.get('window', 64), 
        cfg.get('horizon', 16), 
        train=True,  
        split=cfg.get('split', 0.8), 
        norm_by_train=True
    )
    val_ds = WindowDS(
        s, 
        cfg.get('window', 64), 
        cfg.get('horizon', 16), 
        train=False, 
        split=cfg.get('split', 0.8), 
        norm_by_train=True
    )
    
    return train_ds, val_ds
