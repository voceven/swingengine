# -*- coding: utf-8 -*-
"""
Grandmaster Swing Trading Engine v12 - Modular Components

Modules:
- config.py: Configuration constants and thresholds
- utils.py: Helper functions, Alpaca fetch, ticker utilities
- data_loader.py: TitanDB, HistoryManager
- data_prep.py: Triple-barrier labeling, TCN sequences
- neural.py: SwingTransformer, TCN architectures
- patterns.py: Pattern detection functions (bull flag, phoenix, etc.)
- ml.py: ML training functions (ensemble, CatBoost, TabNet, TCN)
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
from .patterns import (
    detect_bull_flag,
    find_gex_walls,
    detect_downtrend_reversal,
    calculate_flow_factor,
    calculate_solidity_score,
    detect_phoenix_reversal,
    detect_cup_and_handle,
    detect_double_bottom,
    apply_smart_gatekeeper,
)
from .ml import train_ensemble
from .report import generate_full_report
from .transformer import train_hive_mind
