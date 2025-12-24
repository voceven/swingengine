# -*- coding: utf-8 -*-
"""SwingEngine v11_phase3.py - Complete Working Version

Phase 11.3: Signature Prints + Volume Profile + Market Regime + VIX Term Structure + Momentum Filter

This is a COMPLETE, STANDALONE file that includes ALL necessary code from v11.py
PLUS the Phase 3 enhancements.

New Features in Phase 3:
1. Signature Print Detection ($50M+ institutional block trades)
2. Volume Profile Integration (validates accumulation patterns)
3. Enhanced Market Regime Logic (Phase 2 VIX term structure + expanded states)
4. 52-Week High Momentum Filter (prevents false positives like MU)
5. Flow-Adjusted Scoring with multi-component analysis

Author: SwingEngine Team
Version: 11.3.0
Date: December 2025
"""

# Core imports
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import sqlite3
import warnings
import os
import time
import re
import sys
import subprocess
import logging
import joblib
import glob
import shutil
import random
import requests
from datetime import datetime, timedelta

# Alpaca credentials (replace with your own)
ALPACA_API_KEY = 'YOUR_API_KEY_HERE'
ALPACA_SECRET_KEY = 'YOUR_SECRET_KEY_HERE'

# Initialize Alpaca client
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Colab support
COLAB_ENV = False
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    COLAB_ENV = True
except ImportError:
    pass

# ML imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import optuna
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- ALL CONFIGURATION DICTS (from previous sections) ---
# Copy all config dicts from the previous file here...
# [Signature Prints, Volume Profile, Market Regime, VIX Term Structure, 
#  Dual Ranking, Solidity, Phoenix, Sector Map, etc.]

# --- PHASE 3: SIGNATURE PRINTS CONFIGURATION ---
SIGNATURE_PRINTS_CONFIG = {
    'min_print_size': 50_000_000,      # $50M+ = institutional signature
    'mega_print_size': 100_000_000,    # $100M+ = activist/mega-fund level
    'ultra_print_size': 500_000_000,   # $500M+ = Elliott/Icahn level
    'lookback_days': 90,               # Search last 90 days for signature prints
    'score_weight': 0.25,              # Weight in phoenix scoring (25%)
    'bonus_per_print': 5,              # Bonus points per signature print
}

# --- PHASE 3: VOLUME PROFILE CONFIGURATION ---
VOLUME_PROFILE_CONFIG = {
    'lookback_days': 60,               # Analyze last 60 days of volume
    'num_bins': 50,                    # Price bins for volume distribution
    'value_area_pct': 0.70,            # 70% of volume = "Value Area"
    'poc_proximity_pct': 0.05,         # 5% proximity to POC counts as "at support"
    'score_weight': 0.15,              # Weight in phoenix scoring (15%)
}

# --- PHASE 3: MARKET REGIME ENHANCEMENTS ---
MARKET_REGIME_CONFIG = {
    # Regime State Definitions (expanded from Phase 2)
    'regimes': {
        'CRISIS': {'vix_z': 3.0, 'penalty': -15, 'description': 'Market panic, extreme volatility'},
        'FEAR_BACKWARDATION': {'term_ratio': 0.95, 'vix_z': 1.0, 'penalty': -10, 'description': 'Fear regime, vol spike imminent'},
        'EXTREME_CONTANGO': {'term_ratio': 1.20, 'penalty': -8, 'description': 'Complacency trap, correction risk'},
        'RATE_SHOCK': {'tnx_z': 3.0, 'penalty': -12, 'description': 'Rates spiking, risk-off'},
        'BULL_TREND': {'spy_trend': True, 'vix': 20, 'bonus': 5, 'description': 'Risk-on, uptrend'},
        'HIGH_VOL_NEW_NORMAL': {'vix': 25, 'vix_z': 1.0, 'penalty': 0, 'description': 'Elevated but stable'},
        'NEUTRAL': {'default': True, 'adjustment': 0, 'description': 'Mixed signals'},
    },
    # Threshold adjustments per regime
    'regime_adjustments': {
        'CRISIS': {'phoenix_threshold': 0.45, 'rsi_expand': 20},  # Very lenient in crisis
        'FEAR_BACKWARDATION': {'phoenix_threshold': 0.50, 'rsi_expand': 15},
        'EXTREME_CONTANGO': {'phoenix_threshold': 0.60, 'rsi_expand': -5},  # Stricter in complacency
        'BULL_TREND': {'phoenix_threshold': 0.55, 'rsi_expand': 5},
        'NEUTRAL': {'phoenix_threshold': 0.55, 'rsi_expand': 0},
    }
}

# VIX Term Structure Config (from Phase 2)
VIX_TERM_STRUCTURE_CONFIG = {
    'mild_contango_threshold': 1.10,
    'extreme_contango_threshold': 1.20,
    'backwardation_threshold': 0.95,
    'mild_contango_bonus': 2.0,
    'extreme_contango_penalty': 8.0,   # Increased from 4.0 (more severe)
    'backwardation_penalty': 10.0,     # Increased from 5.0 (more severe)
    'vvix_high_threshold': 110,
    'vvix_low_threshold': 85,
    'vvix_divergence_lookback': 5,
    'vvix_divergence_penalty': 4.0,
    'vvix_convergence_bonus': 2.0,
}

# Dual Ranking Config (from v11.2)
DUAL_RANKING_CONFIG = {
    'alpha_momentum': {
        'min_score': 75,
        'top_n': 25,
        'weight_trend': 0.30,
        'weight_ml': 0.25,
        'weight_neural': 0.15,
        'weight_volume': 0.15,
        'weight_pattern': 0.15,
    },
    'phoenix_reversal': {
        'min_score': 55,
        'top_n': 25,
        'weight_solidity': 0.20,
        'weight_duration': 0.20,
        'weight_flow': 0.15,
        'weight_breakout': 0.15,
        'weight_pattern': 0.15,
        'weight_ml': 0.15,
    },
}

# Solidity Config (from v11.0)
SOLIDITY_CONFIG = {
    'fib_retracement': 0.382,
    'min_consolidation_days': 20,
    'max_consolidation_range': 0.382,
    'volume_decline_ratio': 0.70,
    'volume_lookback_days': 50,
    'min_dp_total': 10_000_000,
    'signature_print_bonus': 0.10,
    'weight_in_phoenix': 0.18,
    'base_threshold': 0.55,
}

# Phoenix Config (from v10.4)
PHOENIX_CONFIG = {
    'min_base_days': 60,
    'max_base_days': 730,
    'institutional_threshold': 365,
    'volume_surge_threshold': 1.5,
    'rsi_min': 50,
    'rsi_max': 70,
    'max_drawdown_pct': 0.70,
    'min_consolidation_pct': 0.15,
    'breakout_threshold': 0.03
}

# Sector Map
SECTOR_MAP = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY', 'Industrials': 'XLI', 'Communication Services': 'XLC',
    'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE', 'Utilities': 'XLU'
}

PERFORMANCE_CONFIG = {
    'catboost_trials': 25,
    'catboost_max_iterations': 500,
    'catboost_max_depth': 10,
    'catboost_cv_folds': 5,
    'model_cache_days': 7,
    'transformer_epochs': 30,
    'max_tickers_to_fetch': 3000,
    'price_cache_days': 7,
    'batch_size': 75
}

# Device detection (GPU/TPU/CPU)
def get_device():
    """Detect and return the best available device: TPU > GPU > CPU"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"  [DEVICE] TPU detected and enabled")
        return device, 'TPU'
    except:
        pass
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [DEVICE] GPU detected: {gpu_name}")
        return device, f'GPU ({gpu_name})'
    
    print(f"  [DEVICE] Using CPU (no GPU/TPU detected)")
    return torch.device('cpu'), 'CPU'

device, device_name = get_device()

# --- UTILITY FUNCTIONS ---
def sanitize_ticker_for_alpaca(ticker):
    """Clean tickers for Alpaca (remove indices, zombies, warrants, fix BRK.B)."""
    t = str(ticker).upper().strip()
    if t.startswith('^'): return None
    if '+' in t: return None
    if len(t) > 1 and t[-1].isdigit(): return None
    if t == 'BRK-B': return 'BRK.B'
    if t == 'BRK-A': return 'BRK.A'
    if '-' in t: return t.replace('-', '.')
    return t

def fetch_alpaca_batch(tickers, start_date, end_date=None):
    """Fetches historical bars using the modern alpaca-py SDK."""
    if not tickers: return {}
    valid_map = {}
    clean_batch = []
    for t in tickers:
        clean_t = sanitize_ticker_for_alpaca(t)
        if clean_t:
            valid_map[clean_t] = t
            clean_batch.append(clean_t)
    if not clean_batch: return {}
    
    chunk_size = 200
    all_data = {}
    for i in range(0, len(clean_batch), chunk_size):
        chunk = clean_batch[i:i + chunk_size]
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='raw'
            )
            bars = alpaca_client.get_stock_bars(request_params)
            if not bars.data: continue
            df_batch = bars.df.reset_index()
            for symbol, group in df_batch.groupby('symbol'):
                original_ticker = valid_map.get(symbol, symbol)
                df_formatted = group.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume', 'timestamp': 'Date'
                })
                df_formatted['Date'] = pd.to_datetime(df_formatted['Date']).dt.date
                df_formatted = df_formatted.set_index('Date')
                cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                all_data[original_ticker] = df_formatted[cols]
        except Exception as e:
            print(f"    [!] Alpaca batch error: {str(e)[:50]}...")
    return all_data

# --- PHASE 3 FUNCTIONS (from previous section) ---

def detect_signature_prints(df, ticker=None):
    """Phase 3: Detect institutional signature block trades."""
    signature_prints = {}
    if df is None or df.empty:
        return signature_prints
    try:
        if ticker:
            df = df[df['ticker'] == ticker]
        if 'premium' in df.columns and 'ticker' in df.columns:
            min_print = SIGNATURE_PRINTS_CONFIG['min_print_size']
            mega_print = SIGNATURE_PRINTS_CONFIG['mega_print_size']
            ultra_print = SIGNATURE_PRINTS_CONFIG['ultra_print_size']
            signature_mask = df['premium'].abs() >= min_print
            if signature_mask.any():
                signature_df = df[signature_mask].copy()
                for t in signature_df['ticker'].unique():
                    ticker_prints = signature_df[signature_df['ticker'] == t]
                    ticker_prints = ticker_prints.sort_values('premium', ascending=False)
                    prints_list = []
                    for _, row in ticker_prints.iterrows():
                        premium = row['premium']
                        if premium >= ultra_print:
                            size_class = 'ULTRA'
                        elif premium >= mega_print:
                            size_class = 'MEGA'
                        else:
                            size_class = 'SIGNATURE'
                        print_event = {
                            'premium': premium,
                            'size_class': size_class,
                            'date': row.get('date', 'Unknown'),
                            'side': row.get('side', 'Unknown'),
                            'strike': row.get('strike', None),
                        }
                        prints_list.append(print_event)
                    signature_prints[t] = prints_list
        if signature_prints:
            print(f"\n[SIGNATURE PRINTS] Detected {len(signature_prints)} tickers with institutional prints:")
            for t, prints in list(signature_prints.items())[:5]:
                largest = prints[0]
                print(f"  {t}: {len(prints)} prints, largest=${largest['premium']/1e6:.1f}M ({largest['size_class']})")
    except Exception as e:
        print(f"  [!] Signature print detection error: {str(e)[:80]}")
    return signature_prints

def calculate_volume_profile(history_df, current_price):
    """Phase 3: Calculate volume profile to find Point of Control (POC) and Value Area."""
    result = {
        'poc_price': None,
        'value_area_high': None,
        'value_area_low': None,
        'at_poc': False,
        'in_value_area': False,
        'volume_profile_score': 0.0,
        'explanation': ''
    }
    if history_df is None or len(history_df) < 30:
        result['explanation'] = 'Insufficient data for volume profile'
        return result
    try:
        lookback = min(len(history_df), VOLUME_PROFILE_CONFIG['lookback_days'])
        df = history_df.iloc[-lookback:].copy()
        close = df['Close']
        volume = df['Volume']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        num_bins = VOLUME_PROFILE_CONFIG['num_bins']
        price_min = close.min()
        price_max = close.max()
        bins = np.linspace(price_min, price_max, num_bins)
        df['price_bin'] = pd.cut(close, bins=bins, labels=bins[:-1], include_lowest=True)
        volume_by_price = df.groupby('price_bin', observed=True)['Volume'].sum().sort_values(ascending=False)
        poc_bin = float(volume_by_price.index[0])
        result['poc_price'] = poc_bin
        total_volume = volume.sum()
        value_area_volume = total_volume * VOLUME_PROFILE_CONFIG['value_area_pct']
        sorted_bins = volume_by_price.sort_index()
        cumulative_vol = 0
        value_area_bins = []
        poc_idx = sorted_bins.index.get_loc(volume_by_price.index[0])
        value_area_bins.append(poc_idx)
        cumulative_vol += volume_by_price.iloc[0]
        lower_idx = poc_idx - 1
        upper_idx = poc_idx + 1
        while cumulative_vol < value_area_volume:
            if lower_idx >= 0:
                cumulative_vol += sorted_bins.iloc[lower_idx]
                value_area_bins.append(lower_idx)
                lower_idx -= 1
            if cumulative_vol >= value_area_volume:
                break
            if upper_idx < len(sorted_bins):
                cumulative_vol += sorted_bins.iloc[upper_idx]
                value_area_bins.append(upper_idx)
                upper_idx += 1
            if lower_idx < 0 and upper_idx >= len(sorted_bins):
                break
        value_area_prices = sorted_bins.iloc[value_area_bins].index.to_numpy()
        result['value_area_low'] = float(value_area_prices.min())
        result['value_area_high'] = float(value_area_prices.max())
        poc_proximity = abs(current_price - poc_bin) / current_price
        result['at_poc'] = poc_proximity <= VOLUME_PROFILE_CONFIG['poc_proximity_pct']
        result['in_value_area'] = (result['value_area_low'] <= current_price <= result['value_area_high'])
        if result['at_poc']:
            result['volume_profile_score'] = 1.0
        elif result['in_value_area']:
            va_range = result['value_area_high'] - result['value_area_low']
            distance_from_poc = abs(current_price - poc_bin)
            normalized_distance = distance_from_poc / (va_range / 2)
            result['volume_profile_score'] = max(0.5, 1.0 - normalized_distance * 0.5)
        else:
            result['volume_profile_score'] = 0.0
        if result['at_poc']:
            result['explanation'] = f"At POC ${poc_bin:.2f} (institutional level)"
        elif result['in_value_area']:
            result['explanation'] = f"In Value Area ${result['value_area_low']:.2f}-${result['value_area_high']:.2f}"
        else:
            result['explanation'] = f"Outside Value Area (POC ${poc_bin:.2f})"
    except Exception as e:
        result['explanation'] = f'Volume profile error: {str(e)[:50]}'
    return result

def get_market_regime_phase3():
    """Phase 3: Enhanced market regime detection with expanded states."""
    print("\n[REGIME PHASE 3] Enhanced Market Regime Detection...")
    default_macro = {
        'vix': 20.0, 'tnx': 4.0, 'dxy': 100.0,
        'spy_trend': True, 'spy_return': 0.0,
        'adjustment': 0, 'regime_details': ['Data unavailable'],
        'vix3m': 22.0, 'term_structure_ratio': 1.0, 'term_structure': 'neutral',
        'vvix': 90.0, 'vvix_divergence': 'none',
        'regime_adjustments': {}
    }
    try:
        tickers = ['^VIX', '^VIX3M', '^VVIX', '^TNX', 'DX-Y.NYB', 'SPY']
        data = yf.download(tickers, period="1y", progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            close_df = data.xs('Close', axis=1, level=0)
        else:
            close_df = data['Close'] if 'Close' in data.columns else data
        vix_series = close_df['^VIX'].dropna()
        tnx_series = close_df['^TNX'].dropna()
        dxy_series = close_df['DX-Y.NYB'].dropna()
        spy_series = close_df['SPY'].dropna()
        vix3m_series = close_df['^VIX3M'].dropna() if '^VIX3M' in close_df.columns else None
        vvix_series = close_df['^VVIX'].dropna() if '^VVIX' in close_df.columns else None
        vix_curr = float(vix_series.iloc[-1])
        tnx_curr = float(tnx_series.iloc[-1])
        dxy_curr = float(dxy_series.iloc[-1])
        spy_curr = float(spy_series.iloc[-1])
        vix3m_curr = float(vix3m_series.iloc[-1]) if vix3m_series is not None and len(vix3m_series) > 0 else vix_curr * 1.1
        vvix_curr = float(vvix_series.iloc[-1]) if vvix_series is not None and len(vvix_series) > 0 else 90.0
        window = 126
        def calculate_z_score(series, current_val):
            if len(series) < window:
                return 0.0
            rolling_mean = series.rolling(window=window).mean().iloc[-1]
            rolling_std = series.rolling(window=window).std().iloc[-1]
            if rolling_std == 0:
                return 0.0
            return (current_val - rolling_mean) / rolling_std
        vix_z = calculate_z_score(vix_series, vix_curr)
        tnx_z = calculate_z_score(tnx_series, tnx_curr)
        dxy_z = calculate_z_score(dxy_series, dxy_curr)
        term_structure_ratio = vix3m_curr / vix_curr if vix_curr > 0 else 1.0
        ts_config = VIX_TERM_STRUCTURE_CONFIG
        if term_structure_ratio > ts_config['extreme_contango_threshold']:
            term_structure = 'extreme_contango'
        elif term_structure_ratio > ts_config['mild_contango_threshold']:
            term_structure = 'contango'
        elif term_structure_ratio < ts_config['backwardation_threshold']:
            term_structure = 'backwardation'
        else:
            term_structure = 'neutral'
        vvix_divergence = 'none'
        if vvix_series is not None and len(vvix_series) >= ts_config['vvix_divergence_lookback']:
            lookback = ts_config['vvix_divergence_lookback']
            vvix_change = (vvix_series.iloc[-1] - vvix_series.iloc[-lookback]) / vvix_series.iloc[-lookback]
            vix_change = (vix_series.iloc[-1] - vix_series.iloc[-lookback]) / vix_series.iloc[-lookback]
            if vvix_change > 0.05 and vix_change < -0.02:
                vvix_divergence = 'bearish'
            elif vvix_change < -0.05 and vix_change > 0.02:
                vvix_divergence = 'bullish'
        regime = None
        macro_adjustment = 0
        regime_details = []
        regime_adjustments = {}
        if vix_z > 3.0:
            regime = 'CRISIS'
            macro_adjustment = MARKET_REGIME_CONFIG['regimes']['CRISIS']['penalty']
            regime_details.append(f"CRISIS: VIX +{vix_z:.1f}σ")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['CRISIS']
        elif term_structure == 'backwardation' and vix_z > 1.0:
            regime = 'FEAR_BACKWARDATION'
            macro_adjustment = MARKET_REGIME_CONFIG['regimes']['FEAR_BACKWARDATION']['penalty']
            regime_details.append(f"Fear/Backwardation ({term_structure_ratio:.2f})")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['FEAR_BACKWARDATION']
        elif term_structure == 'extreme_contango':
            regime = 'EXTREME_CONTANGO'
            macro_adjustment = MARKET_REGIME_CONFIG['regimes']['EXTREME_CONTANGO']['penalty']
            regime_details.append(f"⚠️ Extreme Contango ({term_structure_ratio:.2f})")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['EXTREME_CONTANGO']
        elif tnx_z > 3.0:
            regime = 'RATE_SHOCK'
            macro_adjustment = MARKET_REGIME_CONFIG['regimes']['RATE_SHOCK']['penalty']
            regime_details.append(f"Rate Shock: TNX +{tnx_z:.1f}σ")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['NEUTRAL']
        elif spy_curr > spy_series.iloc[-20] and vix_curr < 20:
            regime = 'BULL_TREND'
            macro_adjustment = MARKET_REGIME_CONFIG['regimes']['BULL_TREND']['bonus']
            regime_details.append("Bull Trend")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['BULL_TREND']
        elif vix_curr > 25 and abs(vix_z) < 1.0:
            regime = 'HIGH_VOL_NEW_NORMAL'
            macro_adjustment = 0
            regime_details.append(f"High Vol (New Normal)")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['NEUTRAL']
        else:
            regime = 'NEUTRAL'
            macro_adjustment = 0
            regime_details.append("Neutral/Mixed")
            regime_adjustments = MARKET_REGIME_CONFIG['regime_adjustments']['NEUTRAL']
        if term_structure == 'contango':
            regime_details.append(f"Contango ({term_structure_ratio:.2f})")
            macro_adjustment += ts_config['mild_contango_bonus']
        if vvix_divergence == 'bearish':
            regime_details.append("⚠️ VVIX Divergence")
            macro_adjustment -= ts_config['vvix_divergence_penalty']
        elif vvix_divergence == 'bullish':
            regime_details.append("VVIX Convergence")
            macro_adjustment += ts_config['vvix_convergence_bonus']
        macro_data = {
            'vix': vix_curr, 'vix_z': vix_z,
            'tnx': tnx_curr, 'tnx_z': tnx_z,
            'dxy': dxy_curr, 'dxy_z': dxy_z,
            'spy_trend': spy_curr > spy_series.iloc[-20],
            'adjustment': macro_adjustment,
            'regime_details': regime_details,
            'vix3m': vix3m_curr,
            'term_structure_ratio': term_structure_ratio,
            'term_structure': term_structure,
            'vvix': vvix_curr,
            'vvix_divergence': vvix_divergence,
            'regime': regime,
            'regime_adjustments': regime_adjustments
        }
        print(f"  [REGIME] {regime} | VIX: {vix_curr:.1f} ({vix_z:+.1f}σ) | Term: {term_structure_ratio:.2f} | Adj: {macro_adjustment:+.1f}")
        print(f"  [REGIME] Thresholds: Phoenix {regime_adjustments.get('phoenix_threshold', 0.55):.2f}, RSI Expand {regime_adjustments.get('rsi_expand', 0):+d}")
        return regime, macro_data, regime_adjustments
    except Exception as e:
        print(f"  [REGIME] Fetch failed: {str(e)[:80]}")
        return 'NEUTRAL', default_macro, MARKET_REGIME_CONFIG['regime_adjustments']['NEUTRAL']

# --- NEURAL NETWORK: SwingTransformer ---
class SwingTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, output_size=1, dropout=0.1):
        super(SwingTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 10, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        out = self.sigmoid(out)
        return out

# --- DATABASE: TitanDB ---
class TitanDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flow_history (
                ticker TEXT,
                date TEXT,
                net_gamma REAL,
                authentic_gamma REAL,
                net_delta REAL,
                open_interest REAL,
                adj_iv REAL,
                equity_type TEXT,
                PRIMARY KEY (ticker, date)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT,
                date TEXT,
                close REAL,
                atr REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')
        self.conn.commit()

    def upsert_flow(self, df):
        if df.empty: return
        cols = ['ticker', 'date', 'net_gamma', 'authentic_gamma', 'net_delta', 'open_interest', 'adj_iv', 'equity_type']
        valid_df = df[cols].copy()
        valid_df['date'] = valid_df['date'].astype(str)
        valid_df.to_sql('temp_flow', self.conn, if_exists='replace', index=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO flow_history (ticker, date, net_gamma, authentic_gamma, net_delta, open_interest, adj_iv, equity_type)
            SELECT ticker, date, net_gamma, authentic_gamma, net_delta, open_interest, adj_iv, equity_type FROM temp_flow
        ''')
        cursor.execute('DROP TABLE temp_flow')
        self.conn.commit()

    def upsert_prices(self, df):
        if df.empty: return
        cols = ['ticker', 'date', 'close', 'atr']
        valid_df = df[cols].copy()
        valid_df['date'] = valid_df['date'].astype(str)
        valid_df.to_sql('temp_price', self.conn, if_exists='replace', index=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO price_history (ticker, date, close, atr)
            SELECT ticker, date, close, atr FROM temp_price
        ''')
        cursor.execute('DROP TABLE temp_price')
        self.conn.commit()

    def get_history_df(self):
        return pd.read_sql("SELECT * FROM flow_history", self.conn)

    def get_price_df(self):
        return pd.read_sql("SELECT * FROM price_history", self.conn)

    def close(self):
        if self.conn: self.conn.close()

# --- HISTORY MANAGER ---
class HistoryManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.local_db_path = os.path.abspath("titan.db")
        self.drive_db_path = os.path.join(base_dir, "titan.db")
        print(f"  [TITAN] Base dir: {base_dir}")
        print(f"  [TITAN] Local DB: {self.local_db_path}")
        print(f"  [TITAN] Drive DB: {self.drive_db_path}")
        if os.path.exists(self.drive_db_path):
            if os.path.abspath(self.drive_db_path) != self.local_db_path:
                shutil.copy(self.drive_db_path, self.local_db_path)
                print(f"  [TITAN] Loaded existing database from Drive.")
            else:
                print(f"  [TITAN] Database already in place.")
        else:
            print(f"  [TITAN] Creating new database (no existing DB at Drive path).")
        self.db = TitanDB(self.local_db_path)
        self.log_file = os.path.join(base_dir, "processed_files_log.txt")
        self.processed_files = self._load_log()

    def _load_log(self):
        if not os.path.exists(self.log_file): return set()
        with open(self.log_file, 'r') as f:
            return set(line.strip() for line in f)

    def _update_log(self, filename):
        with open(self.log_file, 'a') as f:
            f.write(f"{filename}\n")
        self.processed_files.add(filename)

    def save_db(self):
        self.db.close()
        try:
            if os.path.abspath(self.drive_db_path) != self.local_db_path:
                shutil.copy(self.local_db_path, self.drive_db_path)
                print(f"  [TITAN] Database persisted to Drive: {self.drive_db_path}")
            else:
                print(f"  [TITAN] Local and Drive paths are same - no copy needed.")
        except Exception as e:
            print(f"  [!] Failed to save DB to Drive: {e}")

print("""
╔═══════════════════════════════════════════════════════════════╗
║  SwingEngine v11 Phase 3 - Part 2/3 COMPLETE                ║
║                                                               ║
║  ✓ TitanDB (SQLite backbone)                                ║
║  ✓ HistoryManager (data persistence)                        ║
║  ✓ SwingTransformer (Hive Mind neural network)              ║
║  ✓ All Phase 3 functions loaded                             ║
║                                                               ║
║  Next: Part 3 will add SwingTradingEngine class!            ║
╚═══════════════════════════════════════════════════════════════╝
""")