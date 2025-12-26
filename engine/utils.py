# -*- coding: utf-8 -*-
"""
Grandmaster Engine v11.5 - Utility Functions

Helper functions for:
- Device detection (GPU/TPU/CPU)
- Ticker sanitization
- Alpaca data fetching
- YFinance session configuration
- Market date utilities
"""

import os
import sys
import subprocess
import requests
import torch
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# ALPACA CLIENT INITIALIZATION
# =============================================================================
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    ALPACA_KEY = os.environ.get('APCA_API_KEY_ID', 'PKZ3P3YMము9N1Y2E9CI')
    ALPACA_SECRET = os.environ.get('APCA_API_SECRET_KEY', 'YucXpMaJIuxv4xKeJXPxLGXk21yZcFVPq3kEXkRi')
    alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    alpaca_client = None
    print("  [WARN] alpaca-py not installed. Run: pip install alpaca-py")

# =============================================================================
# DEVICE DETECTION
# =============================================================================
def get_device():
    """Detect and return the best available device: TPU > GPU > CPU"""
    # Try TPU first (Google Colab TPU runtime)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"  [DEVICE] TPU detected and enabled")
        return device, 'TPU'
    except ImportError:
        pass
    except Exception as e:
        print(f"  [DEVICE] TPU detection failed: {e}")

    # Try GPU (CUDA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [DEVICE] GPU detected: {gpu_name}")
        return device, f'GPU ({gpu_name})'

    # Fallback to CPU
    print(f"  [DEVICE] Using CPU (no GPU/TPU detected)")
    return torch.device('cpu'), 'CPU'

# =============================================================================
# YFINANCE SESSION CONFIGURATION
# =============================================================================
def configure_yfinance_session():
    """
    v10.8: Configure yfinance with browser-like session to avoid throttling.

    Level 1 Defense: User-Agent spoofing + browser headers
    - Makes requests appear to come from Chrome browser
    - Persistent session with connection pooling for efficiency
    """
    session = requests.Session()

    # Chrome 120 on Windows 10 headers (most common browser fingerprint)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    })

    # Connection pooling for efficiency
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=3
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

# Global yfinance session (reused across all downloads)
YF_SESSION = configure_yfinance_session()

# =============================================================================
# MARKET DATE UTILITIES
# =============================================================================
def is_weekend():
    """v10.8: Check if today is weekend (markets closed, no new data)."""
    return datetime.now().weekday() >= 5  # Saturday=5, Sunday=6

def get_market_last_close_date():
    """
    v10.8: Get the most recent market close date.
    Accounts for weekends and common US market holidays.
    """
    today = datetime.now().date()

    # If it's weekend, return last Friday
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:
        # Weekday - market should have data up to yesterday (or today if after 4pm ET)
        return today - timedelta(days=1)

# =============================================================================
# LOGGER CLASS
# =============================================================================
class Logger:
    """Dual output logger - writes to both terminal and file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        try:
            os.fsync(self.log.fileno())
        except:
            pass

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# =============================================================================
# TICKER SANITIZATION
# =============================================================================
def sanitize_ticker_for_alpaca(ticker):
    """Clean tickers for Alpaca (remove indices, zombies, warrants, fix BRK.B)."""
    t = str(ticker).upper().strip()

    # Filter 0: Reject non-ASCII characters (causes latin-1 encoding errors)
    try:
        t.encode('ascii')
    except UnicodeEncodeError:
        return None

    # Filter 1: Exclude Indices and Warrants (containing +)
    if t.startswith('^'):
        return None
    if '+' in t:
        return None  # Warrants often have +, Alpaca rejects them

    # Filter 2: Yahoo "Zombie" Tickers (delisted stocks often get '1' suffix like FTV1)
    if len(t) > 1 and t[-1].isdigit():
        return None

    # Filter 3: Reject empty or too long tickers
    if not t or len(t) > 10:
        return None

    # Map 1: Berkshire Hathaway (Yahoo: BRK-B -> Alpaca: BRK.B)
    if t == 'BRK-B':
        return 'BRK.B'
    if t == 'BRK-A':
        return 'BRK.A'

    # Map 2: General hyphen conversion (Yahoo: ABC-D -> Alpaca: ABC.D)
    if '-' in t:
        return t.replace('-', '.')

    return t

# =============================================================================
# ALPACA DATA FETCHING
# =============================================================================
def fetch_alpaca_batch(tickers, start_date, end_date=None):
    """
    Fetches historical bars using the modern alpaca-py SDK.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date (optional, defaults to now)

    Returns:
        Dictionary of {ticker: DataFrame} with OHLCV data
    """
    if not tickers or not ALPACA_AVAILABLE:
        return {}

    # 1. Sanitize
    valid_map = {}
    clean_batch = []
    for t in tickers:
        clean_t = sanitize_ticker_for_alpaca(t)
        if clean_t:
            valid_map[clean_t] = t
            clean_batch.append(clean_t)

    if not clean_batch:
        return {}

    # 2. Fetch in chunks
    chunk_size = 200
    all_data = {}

    for i in range(0, len(clean_batch), chunk_size):
        chunk = clean_batch[i:i + chunk_size]
        try:
            # Build Request (New SDK format)
            request_params = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='raw'
            )

            # Execute Fetch
            bars = alpaca_client.get_stock_bars(request_params)

            # Process Data
            if not bars.data:
                continue

            # The .df property returns a multi-index dataframe (symbol, timestamp)
            df_batch = bars.df
            df_batch = df_batch.reset_index()

            for symbol, group in df_batch.groupby('symbol'):
                original_ticker = valid_map.get(symbol, symbol)

                # Format to match Engine expectations
                df_formatted = group.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume', 'timestamp': 'Date'
                })

                # Clean up Date column
                df_formatted['Date'] = pd.to_datetime(df_formatted['Date']).dt.date
                df_formatted = df_formatted.set_index('Date')

                # Keep only relevant columns
                cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                all_data[original_ticker] = df_formatted[cols]

        except Exception as e:
            print(f"    [!] Alpaca batch error: {str(e)[:50]}...")

    return all_data

# =============================================================================
# AUTO-INSTALLER
# =============================================================================
def install_requirements():
    """Install required packages if missing."""
    required = [
        'optuna', 'catboost', 'pytorch-tabnet', 'scikit-learn',
        'pandas', 'numpy', 'yfinance', 'torch', 'scipy', 'joblib', 'alpaca-py'
    ]
    for pkg in required:
        try:
            # Handle package names that differ from import names
            import_name = pkg.replace('-', '_')
            if pkg == 'alpaca-py':
                import_name = 'alpaca'
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    'get_device',
    'configure_yfinance_session',
    'YF_SESSION',
    'is_weekend',
    'get_market_last_close_date',
    'Logger',
    'sanitize_ticker_for_alpaca',
    'fetch_alpaca_batch',
    'install_requirements',
    'ALPACA_AVAILABLE',
    'alpaca_client',
]
