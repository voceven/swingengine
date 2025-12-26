# -*- coding: utf-8 -*-
"""
Grandmaster Swing Trading Engine v12 - Modular Components

Modules:
- config.py: Configuration constants and thresholds
- utils.py: Helper functions, Alpaca fetch, ticker utilities
- data_loader.py: TitanDB, HistoryManager
- data_prep.py: Triple-barrier labeling, TCN sequences
- neural.py: SwingTransformer, TCN architectures
"""

from .config import *
from .neural import SwingTransformer, TemporalBlock, TCN
from .utils import (
    get_device,
    configure_yfinance_session,
    YF_SESSION,
    is_weekend,
    get_market_last_close_date,
    Logger,
    sanitize_ticker_for_alpaca,
    fetch_alpaca_batch,
    install_requirements,
    ALPACA_AVAILABLE,
)
from .data_loader import TitanDB, HistoryManager
from .data_prep import triple_barrier_labels, prepare_tcn_sequences
