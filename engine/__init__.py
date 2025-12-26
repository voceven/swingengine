# -*- coding: utf-8 -*-
"""
Grandmaster Swing Trading Engine v11.5 - Modular Components

This package contains the extracted modules from the monolithic v11.py:
- config.py: All configuration constants and thresholds
- utils.py: Helper functions, Alpaca fetch, ticker utilities
- data_loader.py: TitanDB, HistoryManager, file I/O
- neural.py: SwingTransformer, TCN, Hive Mind training
- pattern_detection.py: Bull flags, phoenix, cup-handle, etc.
"""

from .config import *
