# -*- coding: utf-8 -*-
"""SwingEngine_v9_Grandmaster.py

Swing Trading Engine - Version 9.0 (Grandmaster)
Phase 9: Pattern Intelligence + Interpretability

Architecture:
1. BACKBONE: SQLite Database (Scalable History).
2. BRAIN: 5x Ensemble Transformer (Hive Mind) + Tuned CatBoost (Left Brain).
3. CONTEXT: Enhanced Macro-Regime Awareness (VIX, TNX, DXY) with score weighting.
4. EXECUTION: ATR-based Stop Loss & Take Profit.
5. PATTERNS: Bull Flag Detection, GEX Wall Scanner, Downtrend Reversal.
6. INTERPRETABILITY: Human-readable explanation generator for each signal.
7. RISK: Sector capping to prevent over-concentration.

Changelog from v8.4:
- Added bull flag pattern detection (consolidation after strong moves)
- Added GEX wall scanner (gamma exposure support/resistance levels)
- Added downtrend reversal detection (months-long downtrend + DP support)
- Added explanation generator (human-readable reasoning for each pick)
- Added sector capping (max 3 picks per sector per strategy)
- Enhanced macro weighting (DXY/TNX/VIX influence individual scores)
- Removed hard Colab dependency (works locally or in Colab)
"""

import pandas as pd
import numpy as np
import yfinance as yf
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
from datetime import datetime, timedelta

# Colab support (optional - gracefully handles local execution)
COLAB_ENV = False
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    COLAB_ENV = True
except ImportError:
    pass  # Running locally, no Drive mount needed

# --- LOGGER CLASS ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        try: os.fsync(self.log.fileno())
        except: pass

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- AUTO-INSTALLER ---
def install_requirements():
    required = ['optuna', 'catboost', 'scikit-learn', 'pandas', 'numpy', 'yfinance', 'torch']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_requirements()

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
import optuna
from catboost import CatBoostClassifier, Pool
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Transformer uses GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SECTOR_MAP = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY', 'Industrials': 'XLI', 'Communication Services': 'XLC',
    'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE', 'Utilities': 'XLU'
}

# --- PHASE 9 CONFIGURATION ---
# Pattern Detection Thresholds (CALIBRATED after v9.0 run)
BULL_FLAG_CONFIG = {
    'pole_min_gain': 0.08,        # Lowered from 12% to 8% - more realistic for swing trades
    'pole_days': 10,              # Days to measure pole
    'flag_max_range': 0.08,       # Increased from 6% to 8% - allow slightly wider consolidation
    'flag_days': 12,              # Reduced from 15 to 12 - tighter flag window
    'volume_decline_ratio': 0.85  # Relaxed from 0.75 to 0.85 - less strict volume requirement
}

GEX_WALL_CONFIG = {
    'min_support_gamma': 100_000,    # Lowered from 500K - more walls will qualify
    'min_resist_gamma': -100_000,    # Raised from -300K - more walls will qualify
    'proximity_pct': 0.08            # Increased from 5% to 8% - walls further away still count
}

REVERSAL_CONFIG = {
    'lookback_days': 45,          # Reduced from 60 to 45 - shorter downtrend window
    'min_days_below_sma': 20,     # Reduced from 35 to 20 - more realistic in bull market
    'dp_proximity_pct': 0.08      # Increased from 5% to 8% - more DP levels will trigger
}

# Sector Capping (Risk Management)
MAX_PICKS_PER_SECTOR = 3

# Macro Score Weights (how much macro conditions affect individual scores)
MACRO_WEIGHTS = {
    'vix_penalty_threshold': 25,      # VIX above this penalizes scores
    'vix_penalty_per_point': 0.5,     # Score penalty per VIX point above threshold
    'tnx_penalty_threshold': 4.5,     # 10Y yield above this penalizes growth stocks
    'tnx_penalty_per_point': 2.0,     # Score penalty per yield point above threshold
    'dxy_strength_threshold': 104,    # DXY above this indicates strong dollar
    'dxy_penalty_per_point': 0.3      # Score penalty per DXY point above threshold
}

# --- NEURAL NETWORK ARCHITECTURE (TRANSFORMER v2) ---
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

# --- TITAN DATABASE MANAGER (SQLITE BACKBONE) ---
class TitanDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Table: Flow History
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

        # Table: Price History
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
        # Ensure columns match
        cols = ['ticker', 'date', 'net_gamma', 'authentic_gamma', 'net_delta', 'open_interest', 'adj_iv', 'equity_type']
        valid_df = df[cols].copy()
        valid_df['date'] = valid_df['date'].astype(str)

        # Bulk Insert/Replace
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

        # Path Conflict Fix
        if os.path.exists(self.drive_db_path):
            if os.path.abspath(self.drive_db_path) != self.local_db_path:
                shutil.copy(self.drive_db_path, self.local_db_path)
                print(f"  [TITAN] Loaded existing database from Drive.")
            else:
                print(f"  [TITAN] Database already in place.")
        else:
            print(f"  [TITAN] Creating new database.")

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
                print(f"  [TITAN] Database persisted to Drive.")
        except Exception as e:
            print(f"  [!] Failed to save DB to Drive: {e}")

    def sync_history(self, engine, data_folder):
        print(f"\n[HISTORY] Scanning {data_folder} for historical flow data...")
        pattern = os.path.join(data_folder, "bot-eod-report-*.csv")
        found_files = glob.glob(pattern)

        new_files = []
        for f_path in found_files:
            f_name = os.path.basename(f_path)
            if f_name not in self.processed_files:
                try:
                    date_part = f_name.replace("bot-eod-report-", "").replace(".csv", "")
                    datetime.strptime(date_part, '%Y-%m-%d')
                    new_files.append((f_path, f_name, date_part))
                except ValueError: pass

        if new_files:
            print(f"  [HISTORY] Found {len(new_files)} new files. Ingesting into Titan DB...")
            new_files.sort(key=lambda x: x[2])
            for f_path, f_name, date_str in new_files:
                print(f"    Processing {date_str}...")
                df_optimized = engine.optimize_large_dataset(f_path, date_stamp=date_str)
                if not df_optimized.empty:
                    self.db.upsert_flow(df_optimized)
                    self._update_log(f_name)
        else:
            print("  [HISTORY] Flow DB is up to date.")

        self.sync_prices()

    def sync_prices(self):
        print("\n[ORACLE] Syncing Price History & ATR...")
        try:
            hist_df = self.db.get_history_df()
            if hist_df.empty: return
            all_tickers = hist_df['ticker'].unique().tolist()
        except: return

        price_df = self.db.get_price_df()
        existing_tickers = set(price_df['ticker'].unique()) if not price_df.empty else set()

        to_fetch = [t for t in all_tickers if t not in existing_tickers and isinstance(t, str)]

        import random
        refresh_batch = random.sample(list(existing_tickers), int(len(existing_tickers) * 0.10)) if existing_tickers else []
        to_fetch.extend(refresh_batch)
        to_fetch = list(set(to_fetch))

        if not to_fetch:
            print("  [ORACLE] Price DB is up to date.")
            return

        print(f"  [ORACLE] Downloading history for {len(to_fetch)} tickers...")

        new_data = []
        batch_size = 100
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i+batch_size]
            try:
                data = yf.download(batch, period="3mo", group_by='ticker', progress=False, threads=True)
                for t in batch:
                    try:
                        hist = data[t] if len(batch) > 1 else data
                        if not hist.empty:
                            high = hist['High']
                            low = hist['Low']
                            close = hist['Close']
                            tr1 = high - low
                            tr2 = abs(high - close.shift(1))
                            tr3 = abs(low - close.shift(1))
                            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            atr = tr.rolling(14).mean()

                            hist = hist.reset_index()
                            if 'Date' not in hist.columns and 'index' in hist.columns:
                                hist.rename(columns={'index': 'Date'}, inplace=True)

                            atr_vals = atr.values

                            for idx, row in hist.iterrows():
                                d_val = row.get('Date', row.name)
                                close_val = row.get('Close', 0)
                                if isinstance(close_val, pd.Series): close_val = close_val.iloc[0]

                                atr_val = atr_vals[idx] if idx < len(atr_vals) else 0
                                if np.isnan(atr_val): atr_val = 0

                                d_str = str(d_val).split()[0]
                                new_data.append({'ticker': t, 'date': d_str, 'close': float(close_val), 'atr': float(atr_val)})
                    except: pass
            except: pass

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.db.upsert_prices(new_df)
            print(f"  [ORACLE] Price DB updated with {len(new_data)} records.")

class SwingTradingEngine:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.model_path = os.path.join(self.base_dir, "grandmaster_cat_v8.pkl")
        self.transformer_path = os.path.join(self.base_dir, "grandmaster_transformer_v8.pth")

        self.scaler = StandardScaler()
        self.nn_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.features_list = []
        self.full_df = pd.DataFrame()
        self.price_cache_file = os.path.join(self.base_dir, "price_cache_v79.csv")

        self.model = None
        self.nn_model = None
        self.model_trained = False

        self.spy_metrics = {'return': 0.0, 'rsi': 50.0, 'trend': 0.0}
        self.sector_data = {}
        self.market_breadth = 50.0
        self.market_regime = "Neutral"
        self.macro_data = {'vix': 20, 'tnx': 4.0, 'dxy': 100, 'spy_trend': True, 'spy_return': 0, 'adjustment': 0, 'regime_details': []}
        self.optimized_bot_file = os.path.join(self.base_dir, "optimized_bot_data_v62.csv")
        self.earnings_map = {}
        self.sector_map_local = {}
        self.cap_map = {}
        self.dp_support_levels = {}
        self.price_history_cache = {}  # Cache for pattern detection
        self.strike_gamma_data = {}    # Strike-level gamma for GEX wall detection

        self.history_mgr = HistoryManager(self.base_dir)

    # --- UTILITIES ---
    def normalize_ticker(self, ticker):
        t = str(ticker).upper().strip().rstrip('=')
        indices_map = {'SPX': '^SPX', 'VIX': '^VIX', 'RUT': '^RUT', 'DJX': '^DJX', 'NDX': '^NDX'}
        if t in indices_map: return indices_map[t]
        t = t.replace('.', '-')
        if t == 'BRKB': return 'BRK-B'
        if len(t) > 4 and t[-1].isdigit(): t = re.sub(r'\d+$', '', t)
        return t

    def safe_read(self, filepath, name):
        if not filepath or not os.path.exists(filepath): return pd.DataFrame()
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            print(f"  [+] Loaded {name}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"  [!] Error reading {name}: {e}")
            return pd.DataFrame()

    def get_market_regime(self):
        print("\n[TITAN] Assessing Macro Regime...")
        try:
            tickers = ['^VIX', '^TNX', 'DX-Y.NYB', 'SPY']
            data = yf.download(tickers, period="5d", progress=False)['Close']
            vix = data['^VIX'].iloc[-1]
            tnx = data['^TNX'].iloc[-1]
            dxy = data['DX-Y.NYB'].iloc[-1]
            spy_current = data['SPY'].iloc[-1]
            spy_start = data['SPY'].iloc[0]
            spy_trend = spy_current > spy_start
            spy_return = (spy_current - spy_start) / spy_start

            # Store raw macro values for score weighting
            self.macro_data = {
                'vix': float(vix),
                'tnx': float(tnx),
                'dxy': float(dxy),
                'spy_trend': spy_trend,
                'spy_return': float(spy_return)
            }

            regime = "Neutral"
            regime_details = []

            if vix > 30:
                regime = "Bear Volatility"
                regime_details.append(f"High fear (VIX {vix:.1f})")
            elif vix > 20 and not spy_trend:
                regime = "Correction"
                regime_details.append(f"Elevated volatility + weakness")
            elif tnx > 4.5 and dxy > 105:
                regime = "Rates Pressure"
                regime_details.append(f"Yields pressuring equities (TNX {tnx:.2f}%)")
            elif spy_trend and vix < 20:
                regime = "Bull Trend"
                regime_details.append(f"Risk-on environment")
            else:
                regime_details.append("Mixed signals")

            # Calculate macro adjustment score (used to weight individual picks)
            macro_adjustment = 0
            if vix > MACRO_WEIGHTS['vix_penalty_threshold']:
                macro_adjustment -= (vix - MACRO_WEIGHTS['vix_penalty_threshold']) * MACRO_WEIGHTS['vix_penalty_per_point']
            if tnx > MACRO_WEIGHTS['tnx_penalty_threshold']:
                macro_adjustment -= (tnx - MACRO_WEIGHTS['tnx_penalty_threshold']) * MACRO_WEIGHTS['tnx_penalty_per_point']
            if dxy > MACRO_WEIGHTS['dxy_strength_threshold']:
                macro_adjustment -= (dxy - MACRO_WEIGHTS['dxy_strength_threshold']) * MACRO_WEIGHTS['dxy_penalty_per_point']

            self.macro_data['adjustment'] = macro_adjustment
            self.macro_data['regime_details'] = regime_details

            print(f"  [MACRO] VIX: {vix:.2f} | TNX: {tnx:.2f}% | DXY: {dxy:.2f} | Regime: {regime}")
            print(f"  [MACRO] Score Adjustment: {macro_adjustment:+.1f} points")
            self.market_regime = regime
            return regime
        except Exception as e:
            print(f"  [MACRO] Data fetch failed ({e}). Defaulting to Neutral.")
            self.macro_data = {'vix': 20, 'tnx': 4.0, 'dxy': 100, 'spy_trend': True, 'spy_return': 0, 'adjustment': 0, 'regime_details': ['Data unavailable']}
            return "Neutral"

    def optimize_large_dataset(self, big_filepath, date_stamp=None):
        chunk_size = 200000
        chunks = []
        strike_chunks = []  # Preserve strike-level data for GEX analysis
        iv_factor = None
        try:
            for chunk in pd.read_csv(big_filepath, chunksize=chunk_size, low_memory=False):
                chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_')
                if iv_factor is None and 'implied_volatility' in chunk.columns:
                    sample = chunk['implied_volatility'].dropna().head(100)
                    iv_factor = 100.0 if not sample.empty and sample.mean() < 5.0 else 1.0

                mask = pd.Series(False, index=chunk.index)
                if 'gamma' in chunk.columns: mask |= (chunk['gamma'].abs() > 0.001)
                if 'premium' in chunk.columns: mask |= (chunk['premium'] > 170000)
                chunk = chunk[mask].copy()
                if chunk.empty: continue

                if 'underlying_symbol' in chunk.columns:
                     chunk['ticker'] = chunk['underlying_symbol'].apply(self.normalize_ticker)

                chunk['weight'] = 0.0
                if 'side' in chunk.columns:
                    chunk.loc[chunk['side'].str.upper().isin(['ASK', 'A', 'BUY']), 'weight'] = 1.0
                    chunk.loc[chunk['side'].str.upper().isin(['BID', 'B', 'SELL']), 'weight'] = -0.5

                mult = chunk['size'] if 'size' in chunk.columns else 100
                if 'gamma' in chunk.columns: chunk['net_gamma'] = chunk['gamma'] * mult * chunk['weight']
                else: chunk['net_gamma'] = 0

                vol = chunk['volume'].fillna(0)
                oi = chunk['open_interest'].fillna(0)
                chunk['is_authentic'] = vol > oi
                chunk['authentic_gamma'] = np.where(chunk['is_authentic'], chunk['net_gamma'], 0)

                if 'delta' in chunk.columns: chunk['net_delta'] = chunk['delta'] * mult * chunk['weight']
                else: chunk['net_delta'] = 0

                if 'implied_volatility' in chunk.columns: chunk['adj_iv'] = chunk['implied_volatility'] * iv_factor
                else: chunk['adj_iv'] = 0

                cols = ['ticker', 'net_gamma', 'authentic_gamma', 'net_delta', 'open_interest', 'adj_iv', 'equity_type']
                if 'sector' in chunk.columns: cols.append('sector')
                for c in cols:
                    if c not in chunk.columns: chunk[c] = 0 if c != 'equity_type' else 'Unknown'
                chunks.append(chunk[cols])

                # PRESERVE STRIKE-LEVEL DATA for GEX wall detection
                if 'strike' in chunk.columns and 'ticker' in chunk.columns:
                    strike_data = chunk[['ticker', 'strike', 'net_gamma']].copy()
                    strike_data = strike_data[strike_data['net_gamma'].abs() > 100]  # Filter significant gamma
                    if not strike_data.empty:
                        strike_chunks.append(strike_data)

            full_optimized = pd.concat(chunks, ignore_index=True)
            agg_rules = {'net_gamma': 'sum', 'authentic_gamma': 'sum', 'net_delta': 'sum', 'open_interest': 'sum', 'adj_iv': 'mean', 'equity_type': 'first'}
            if 'sector' in full_optimized.columns: agg_rules['sector'] = 'first'
            final_df = full_optimized.groupby('ticker').agg(agg_rules).reset_index()

            # Store strike-level gamma data for GEX analysis
            if strike_chunks and date_stamp is None:  # Only for today's data
                strike_df = pd.concat(strike_chunks, ignore_index=True)
                # Aggregate by ticker+strike
                strike_agg = strike_df.groupby(['ticker', 'strike'])['net_gamma'].sum().reset_index()
                # Store as dict: ticker -> {strike: gamma}
                for ticker in strike_agg['ticker'].unique():
                    ticker_strikes = strike_agg[strike_agg['ticker'] == ticker]
                    self.strike_gamma_data[ticker] = dict(zip(ticker_strikes['strike'], ticker_strikes['net_gamma']))
                print(f"  [GEX] Preserved strike-level gamma for {len(self.strike_gamma_data)} tickers")

            if date_stamp: final_df['date'] = date_stamp
            elif not date_stamp: final_df.to_csv(self.optimized_bot_file, index=False)
            return final_df
        except Exception as e:
            print(f"  [!] Dataset optimization error: {e}")
            return pd.DataFrame()

    def generate_temporal_features(self, current_flow_df):
        print("\n[2.5/4] Calculating Temporal Features (Velocity)...")
        hist_df = self.history_mgr.db.get_history_df()
        if hist_df.empty: return current_flow_df

        today_str = datetime.now().strftime('%Y-%m-%d')
        hist_df = hist_df[hist_df['date'] != today_str]

        today_df = current_flow_df.copy()
        today_df['date'] = today_str

        cols = ['ticker', 'date', 'net_gamma', 'authentic_gamma', 'net_delta', 'open_interest']
        valid_cols = [c for c in cols if c in hist_df.columns and c in today_df.columns]
        if len(valid_cols) < 3: return current_flow_df

        combined = pd.concat([hist_df[valid_cols], today_df[valid_cols]])
        try:
            combined['date'] = pd.to_datetime(combined['date'])
            combined = combined.sort_values('date')
            gamma_pivot = combined.pivot_table(index='ticker', columns='date', values='net_gamma', aggfunc='sum').fillna(0)
            gamma_velocity = gamma_pivot.diff(axis=1).mean(axis=1).rename('gamma_velocity')
            oi_pivot = combined.pivot_table(index='ticker', columns='date', values='open_interest', aggfunc='sum').fillna(0)
            oi_accel = oi_pivot.diff(axis=1).mean(axis=1).rename('oi_accel')
            features_df = pd.concat([gamma_velocity, oi_accel], axis=1).reset_index()
            updated_df = pd.merge(current_flow_df, features_df, on='ticker', how='left').fillna(0)
            print(f"  [+] Generated Velocity Features for {len(updated_df)} tickers.")
            return updated_df
        except: return current_flow_df

    # --- SUPERVISED LEARNING PIPELINE ---
    def prepare_supervised_data(self, window_size=3, lookahead=1):
        print(f"\n[TRANSFORMER] Preparing Supervised Data (Window={window_size}, Lookahead={lookahead}d)...")
        hist_df = self.history_mgr.db.get_history_df()
        if hist_df.empty: return None, None, None, None

        needed_cols = ['ticker', 'date', 'net_gamma', 'net_delta', 'open_interest', 'adj_iv']
        valid_cols = [c for c in needed_cols if c in hist_df.columns]
        if len(valid_cols) < 4: return None, None, None, None

        price_df = self.history_mgr.db.get_price_df()
        if price_df.empty: return self.prepare_inference_data(window_size)

        price_df['date'] = pd.to_datetime(price_df['date'])

        df = hist_df[valid_cols].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])

        feature_cols = [c for c in valid_cols if c not in ['ticker', 'date']]
        df[feature_cols] = self.nn_scaler.fit_transform(df[feature_cols])

        sequences = []
        targets = []
        tickers_list = []

        grouped = df.groupby('ticker')
        price_lookup = price_df.set_index(['ticker', 'date'])['close'].to_dict()
        count_labeled = 0

        print(f"  [TRANSFORMER] Building sequences from {len(df)} rows...")

        for ticker, group in grouped:
            if len(group) >= window_size:
                for i in range(len(group) - window_size):
                    seq_slice = group.iloc[i : i+window_size]
                    seq_dates = seq_slice['date'].values
                    last_date = pd.to_datetime(seq_dates[-1])
                    target_date = last_date + timedelta(days=lookahead)
                    start_price = price_lookup.get((ticker, last_date))
                    end_price = None
                    for d in range(3):
                        check_date = target_date + timedelta(days=d)
                        if (ticker, check_date) in price_lookup:
                            end_price = price_lookup[(ticker, check_date)]
                            break
                    if start_price and end_price and start_price > 0:
                        ret = (end_price - start_price) / start_price
                        label = 1 if ret > 0.01 else 0
                        sequences.append(seq_slice[feature_cols].values)
                        targets.append(label)
                        count_labeled += 1

                sequences.append(group[feature_cols].values[-window_size:])
                targets.append(-1)
                tickers_list.append(ticker)

        if not sequences: return None, None, None, None
        X = np.array(sequences)
        y = np.array(targets)
        print(f"  [TRANSFORMER] Created {len(X)} sequences. Labeled Training Data: {count_labeled}.")
        return X, y, tickers_list, feature_cols

    def prepare_inference_data(self, window_size):
        hist_df = self.history_mgr.db.get_history_df()
        if hist_df.empty: return None, None, None, None

        needed_cols = ['ticker', 'date', 'net_gamma', 'net_delta', 'open_interest', 'adj_iv']
        valid_cols = [c for c in needed_cols if c in hist_df.columns]
        if len(valid_cols) < 4: return None, None, None, None
        df = hist_df[valid_cols].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        feature_cols = [c for c in valid_cols if c not in ['ticker', 'date']]
        df[feature_cols] = self.nn_scaler.fit_transform(df[feature_cols])
        sequences, targets, tickers_list = [], [], []
        for ticker, group in df.groupby('ticker'):
            if len(group) >= window_size:
                sequences.append(group[feature_cols].values[-window_size:])
                targets.append(-1)
                tickers_list.append(ticker)
        if not sequences: return None, None, None, None
        return np.array(sequences), np.array(targets), tickers_list, feature_cols

    # --- HIVE MIND (ENSEMBLE) ---
    def train_run_transformer(self):
        X_data, y_data, inference_tickers, feature_cols = self.prepare_supervised_data(window_size=3, lookahead=1)
        if X_data is None or np.sum(y_data != -1) == 0:
             X_data, y_data, inference_tickers, feature_cols = self.prepare_supervised_data(window_size=2, lookahead=1)

        if X_data is None: return

        train_mask = y_data != -1
        X_train = X_data[train_mask]
        y_train = y_data[train_mask]
        X_infer = X_data[~train_mask]

        input_size = len(feature_cols)
        num_models = 5
        ensemble_preds = []

        if len(X_train) > 10:
            print(f"  [HIVE MIND] Training {num_models} Independent Transformers (Bagging)...")
            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            X_infer_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device) if len(X_infer) > 0 else None

            for i in range(num_models):
                print(f"    ... Training Brain #{i+1} ...")
                torch.manual_seed(42 + i)
                model = SwingTransformer(input_size=input_size, d_model=128, num_layers=3).to(device)
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                criterion = nn.BCELoss()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                model.eval()
                if X_infer_tensor is not None:
                    with torch.no_grad():
                        preds = model(X_infer_tensor).cpu().numpy().flatten()
                        ensemble_preds.append(preds)
        else: return

        if ensemble_preds:
            avg_preds = np.mean(ensemble_preds, axis=0)
            nn_df = pd.DataFrame({'ticker': inference_tickers, 'nn_score': avg_preds})
            min_s, max_s = nn_df['nn_score'].min(), nn_df['nn_score'].max()
            if max_s - min_s > 0.0001:
                nn_df['nn_score'] = (nn_df['nn_score'] - min_s) / (max_s - min_s) * 100
            else: nn_df['nn_score'] = nn_df['nn_score'] * 100
            if not self.full_df.empty:
                self.full_df = pd.merge(self.full_df, nn_df, on='ticker', how='left').fillna(0)

    # --- RESTORED SECTOR FETCH ---
    def fetch_sector_history(self):
        print("  [CONTEXT] Fetching Sector ETF History...")
        try:
            etfs = list(SECTOR_MAP.values()) + ['SPY']
            data = yf.download(etfs, period="1mo", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            for etf in etfs:
                if etf in data.columns:
                    series = data[etf].dropna()
                    if len(series) > 1:
                        self.sector_data[etf] = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
                    else: self.sector_data[etf] = 0.0
        except: pass

    # --- RESTORED BREADTH CALC ---
    def calculate_market_breadth(self, cached_data):
        print("  [MACRO] Calculating Market Breadth...")
        if not cached_data: return 50.0
        count = sum(1 for t, m in cached_data.items() if m.get('dist_sma50', 0) > 0)
        return (count / len(cached_data) * 100) if cached_data else 50.0

    # --- RESTORED FUNDAMENTAL ANALYSIS (CRITICAL FIX) ---
    def analyze_fundamentals_and_sector(self, ticker, equity_type):
        res = {'quality_score': 0, 'sector_status': 'Neutral', 'quality_label': 'Unknown', 'ambush_bonus': 0}

        # ETF Filter
        if str(equity_type).upper() == 'ETF':
            res['quality_label'] = 'ETF'
        else:
            # Quality Check
            mkt_cap = self.cap_map.get(ticker, 0)
            if mkt_cap > 10_000_000_000:
                res['quality_score'] = 3
                res['quality_label'] = 'Quality Leader'
            elif mkt_cap > 2_000_000_000:
                res['quality_score'] = 0
                res['quality_label'] = 'Standard'
            else:
                res['quality_score'] = 1
                res['quality_label'] = 'Speculative'

        # Sector Check
        sector_name = self.sector_map_local.get(ticker, 'Unknown')
        etf = SECTOR_MAP.get(sector_name, None)
        if etf and etf in self.sector_data and 'SPY' in self.sector_data:
            sector_perf = self.sector_data[etf]
            spy_perf = self.sector_data['SPY']

            if sector_perf > spy_perf:
                res['sector_status'] = 'Leading Sector'
                res['quality_score'] += 4
                res['ambush_bonus'] = -10
            else:
                res['sector_status'] = 'Lagging Sector'
                res['quality_score'] -= 2
                res['ambush_bonus'] = 5
        return res

    # --- RESTORED EARNINGS CHECK ---
    def check_earnings_proximity(self, ticker):
        if ticker in self.earnings_map:
            next_date_str = self.earnings_map[ticker]
            try:
                earnings_ts = pd.to_datetime(next_date_str)
                days_away = (earnings_ts - pd.Timestamp.now()).days
                if 0 <= days_away <= 5: return True
                return False
            except: pass
        return False

    # --- PHASE 9: PATTERN DETECTION METHODS ---

    def detect_bull_flag(self, ticker, history_df):
        """
        Detect bull flag pattern: strong upward move (pole) followed by consolidation (flag).

        Bull flag criteria:
        1. Strong upward move (pole): 12%+ gain in first ~10 days
        2. Consolidation (flag): <6% range in next 10-20 days
        3. Volume declining during flag phase

        Returns dict with pattern detection results and explanation.
        """
        result = {
            'is_flag': False,
            'flag_score': 0.0,
            'pole_gain': 0.0,
            'flag_range': 0.0,
            'explanation': ''
        }

        lookback = BULL_FLAG_CONFIG['pole_days'] + BULL_FLAG_CONFIG['flag_days'] + 5

        if history_df is None or len(history_df) < lookback:
            result['explanation'] = 'Insufficient price history for pattern detection'
            return result

        try:
            close = history_df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            volume = history_df['Volume'] if 'Volume' in history_df.columns else None
            if isinstance(volume, pd.DataFrame) and volume is not None:
                volume = volume.iloc[:, 0]

            # Use last N days of data
            close = close.iloc[-lookback:]
            if volume is not None:
                volume = volume.iloc[-lookback:]

            pole_days = BULL_FLAG_CONFIG['pole_days']
            flag_days = BULL_FLAG_CONFIG['flag_days']

            # Pole phase: first segment of the lookback
            pole_start = close.iloc[0]
            pole_end = close.iloc[pole_days]
            pole_gain = (pole_end - pole_start) / pole_start if pole_start > 0 else 0

            # Flag phase: last segment (consolidation)
            flag_segment = close.iloc[-flag_days:]
            flag_high = flag_segment.max()
            flag_low = flag_segment.min()
            flag_range = (flag_high - flag_low) / flag_segment.mean() if flag_segment.mean() > 0 else 1

            # Volume analysis (optional but improves accuracy)
            volume_declining = True
            if volume is not None and len(volume) >= lookback:
                pole_volume = volume.iloc[:pole_days].mean()
                flag_volume = volume.iloc[-flag_days:].mean()
                volume_declining = flag_volume < pole_volume * BULL_FLAG_CONFIG['volume_decline_ratio']

            # Check if pattern matches
            is_flag = (
                pole_gain >= BULL_FLAG_CONFIG['pole_min_gain'] and
                flag_range <= BULL_FLAG_CONFIG['flag_max_range'] and
                volume_declining
            )

            result['pole_gain'] = pole_gain
            result['flag_range'] = flag_range
            result['is_flag'] = is_flag

            if is_flag:
                result['flag_score'] = min(1.0, (pole_gain / 0.15) * 0.5 + (1 - flag_range / 0.06) * 0.5)
                result['explanation'] = f"BULL FLAG: {pole_gain*100:.1f}% pole gain, {flag_range*100:.1f}% consolidation"
                if volume_declining:
                    result['explanation'] += ", declining volume"
            elif pole_gain > 0.08:
                # Partial pattern - might be forming
                result['flag_score'] = 0.3
                result['explanation'] = f"Potential flag forming: {pole_gain*100:.1f}% move, watching consolidation"
            else:
                result['explanation'] = 'No bull flag pattern detected'

        except Exception as e:
            result['explanation'] = f'Pattern detection error: {str(e)[:50]}'

        return result

    def find_gex_walls(self, ticker, current_price, bot_df=None):
        """
        Find gamma exposure (GEX) walls that could act as support/resistance.

        GEX walls are strike prices with concentrated gamma that create
        "magnetic" levels where market makers' hedging activity provides support or resistance.

        Returns dict with wall levels and explanation.
        """
        result = {
            'support_wall': None,
            'resistance_wall': None,
            'wall_protection_score': 0.0,
            'explanation': ''
        }

        if current_price is None or current_price <= 0:
            result['explanation'] = 'Invalid price data'
            return result

        # PRIORITY 1: Use cached strike-level gamma data
        if ticker in self.strike_gamma_data and self.strike_gamma_data[ticker]:
            try:
                strike_gamma = self.strike_gamma_data[ticker]  # dict: {strike: gamma}

                # Support walls: strikes below price with large positive gamma
                support_candidates = {k: v for k, v in strike_gamma.items()
                                     if k < current_price and v > GEX_WALL_CONFIG['min_support_gamma']}

                if support_candidates:
                    best_strike = max(support_candidates.keys(), key=lambda x: support_candidates[x])
                    result['support_wall'] = float(best_strike)
                    support_gamma = support_candidates[best_strike]
                    proximity = (current_price - best_strike) / current_price

                    if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                        result['wall_protection_score'] = min(1.0, support_gamma / 500_000)

                # Resistance walls: strikes above price with large negative gamma
                resist_candidates = {k: v for k, v in strike_gamma.items()
                                    if k > current_price and v < GEX_WALL_CONFIG['min_resist_gamma']}

                if resist_candidates:
                    best_resist = min(resist_candidates.keys(), key=lambda x: resist_candidates[x])
                    result['resistance_wall'] = float(best_resist)

                # Build explanation
                explanations = []
                if result['support_wall']:
                    support_dist = (current_price - result['support_wall']) / current_price * 100
                    gamma_val = support_candidates.get(result['support_wall'], 0)
                    explanations.append(f"GEX support ${result['support_wall']:.0f} ({support_dist:.1f}% below, {gamma_val/1000:.0f}K gamma)")
                if result['resistance_wall']:
                    resist_dist = (result['resistance_wall'] - current_price) / current_price * 100
                    explanations.append(f"GEX resist ${result['resistance_wall']:.0f} ({resist_dist:.1f}% above)")

                result['explanation'] = " | ".join(explanations) if explanations else "No significant GEX walls"
                return result

            except Exception as e:
                result['explanation'] = f'GEX cache error: {str(e)[:30]}'

        # PRIORITY 2: Check if we have strike-level data in bot_df
        if bot_df is not None and not bot_df.empty and 'strike' in bot_df.columns:
            try:
                ticker_data = bot_df[bot_df['ticker'] == ticker] if 'ticker' in bot_df.columns else bot_df

                if ticker_data.empty or 'net_gamma' not in ticker_data.columns:
                    result['explanation'] = 'No gamma data for strike analysis'
                    return result

                # Aggregate gamma by strike
                strike_gamma = ticker_data.groupby('strike')['net_gamma'].sum()

                # Support walls: strikes below price with large positive gamma
                support_mask = (strike_gamma.index < current_price) & (strike_gamma > GEX_WALL_CONFIG['min_support_gamma'])
                support_candidates = strike_gamma[support_mask]

                if not support_candidates.empty:
                    result['support_wall'] = float(support_candidates.idxmax())
                    support_gamma = support_candidates.max()
                    proximity = (current_price - result['support_wall']) / current_price

                    if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                        result['wall_protection_score'] = min(1.0, support_gamma / 500_000)

                # Resistance walls: strikes above price with large negative gamma
                resist_mask = (strike_gamma.index > current_price) & (strike_gamma < GEX_WALL_CONFIG['min_resist_gamma'])
                resist_candidates = strike_gamma[resist_mask]

                if not resist_candidates.empty:
                    result['resistance_wall'] = float(resist_candidates.idxmin())

                # Build explanation
                explanations = []
                if result['support_wall']:
                    support_dist = (current_price - result['support_wall']) / current_price * 100
                    explanations.append(f"GEX support at ${result['support_wall']:.2f} ({support_dist:.1f}% below)")
                if result['resistance_wall']:
                    resist_dist = (result['resistance_wall'] - current_price) / current_price * 100
                    explanations.append(f"GEX resistance at ${result['resistance_wall']:.2f} ({resist_dist:.1f}% above)")

                result['explanation'] = " | ".join(explanations) if explanations else "No significant GEX walls"
                return result

            except Exception as e:
                result['explanation'] = f'GEX analysis error: {str(e)[:50]}'

        # PRIORITY 3: Fallback to dark pool support levels
        if ticker in self.dp_support_levels:
            dp_levels = self.dp_support_levels[ticker]
            support_levels = [p for p in dp_levels if p < current_price]
            if support_levels:
                best_support = max(support_levels)
                proximity = (current_price - best_support) / current_price
                if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                    result['support_wall'] = best_support
                    result['wall_protection_score'] = 0.5  # DP support is less reliable than GEX
                    result['explanation'] = f"DP support at ${best_support:.2f} ({proximity*100:.1f}% below)"
                else:
                    result['explanation'] = f"DP support at ${best_support:.2f} (too far: {proximity*100:.1f}%)"
            else:
                result['explanation'] = 'No support levels detected'
        else:
            result['explanation'] = 'No strike-level or DP data available'

        return result

    def detect_downtrend_reversal(self, ticker, history_df):
        """
        Detect potential reversal setup: months-long downtrend with dark pool support at bottom.

        Criteria:
        1. Extended downtrend: price below 50 SMA for 35+ of last 60 days
        2. Dark pool accumulation near current price levels
        3. RSI showing potential bullish divergence

        Returns dict with reversal detection results and explanation.
        """
        result = {
            'is_reversal': False,
            'reversal_score': 0.0,
            'days_below_sma': 0,
            'has_dp_support': False,
            'explanation': ''
        }

        lookback = REVERSAL_CONFIG['lookback_days']

        if history_df is None or len(history_df) < lookback:
            result['explanation'] = 'Insufficient history for reversal analysis'
            return result

        try:
            close = history_df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            # Calculate SMA50
            sma50 = close.rolling(50).mean()

            # Count days below SMA50 in lookback period
            recent_close = close.iloc[-lookback:]
            recent_sma = sma50.iloc[-lookback:]
            days_below = (recent_close < recent_sma).sum()
            result['days_below_sma'] = int(days_below)

            is_downtrend = days_below >= REVERSAL_CONFIG['min_days_below_sma']

            # Check for dark pool support at current levels
            current_price = float(close.iloc[-1])
            has_dp_support = False
            dp_level = None

            if ticker in self.dp_support_levels:
                dp_levels = self.dp_support_levels[ticker]
                nearby_support = [p for p in dp_levels if abs(p - current_price) / current_price < REVERSAL_CONFIG['dp_proximity_pct']]
                if nearby_support:
                    has_dp_support = True
                    dp_level = max(nearby_support)

            result['has_dp_support'] = has_dp_support

            # Check for RSI bullish divergence (price making lower lows, RSI making higher lows)
            has_divergence = False
            if len(close) > 20:
                # Simple divergence check
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-9)
                rsi = 100 - (100 / (1 + rs))

                # Compare first half vs second half of recent period
                price_trend = close.iloc[-10:].mean() < close.iloc[-20:-10].mean()  # Price declining
                rsi_trend = rsi.iloc[-10:].mean() > rsi.iloc[-20:-10].mean()  # RSI rising
                has_divergence = price_trend and rsi_trend

            # Determine if reversal setup is present
            is_reversal = is_downtrend and (has_dp_support or has_divergence)
            result['is_reversal'] = is_reversal

            if is_reversal:
                score_components = []
                base_score = 0.5

                if has_dp_support:
                    base_score += 0.3
                    score_components.append(f"DP support at ${dp_level:.2f}")
                if has_divergence:
                    base_score += 0.2
                    score_components.append("RSI bullish divergence")

                result['reversal_score'] = min(1.0, base_score)
                result['explanation'] = f"REVERSAL SETUP: {days_below}/{lookback} days in downtrend, " + ", ".join(score_components)
            elif is_downtrend:
                result['reversal_score'] = 0.2
                result['explanation'] = f"Downtrend ({days_below}/{lookback} days below SMA50) but no support confirmation"
            else:
                result['explanation'] = 'Not in extended downtrend'

        except Exception as e:
            result['explanation'] = f'Reversal detection error: {str(e)[:50]}'

        return result

    def generate_signal_explanation(self, row, patterns):
        """
        Generate human-readable explanation for why this ticker is being recommended.
        Combines macro context, flow analysis, pattern recognition, and AI confidence.

        Returns a structured explanation string.
        """
        reasons = []
        warnings = []

        # 1. Macro Context
        if hasattr(self, 'macro_data') and self.macro_data:
            macro = self.macro_data
            reasons.append(f"Macro: {self.market_regime}")
            if macro.get('adjustment', 0) < -5:
                warnings.append(f"Macro headwinds ({macro['adjustment']:+.1f} adjustment)")

        # 2. Flow Analysis
        net_gamma = row.get('net_gamma', 0)
        gamma_velocity = row.get('gamma_velocity', 0)
        dp_sentiment = row.get('dp_sentiment', 0)
        dp_total = row.get('dp_total', 0)

        if net_gamma > 100:
            reasons.append(f"Bullish gamma (+{net_gamma:.0f})")
        elif net_gamma < -100:
            warnings.append(f"Bearish gamma ({net_gamma:.0f})")

        if gamma_velocity > 30:
            reasons.append(f"Accelerating gamma ({gamma_velocity:.0f}% velocity)")

        if dp_sentiment > 0.3:
            reasons.append("Dark pool accumulation")
        elif dp_sentiment < -0.3:
            warnings.append("Dark pool distribution")

        if dp_total > 5_000_000:
            reasons.append(f"Heavy DP activity (${dp_total/1e6:.1f}M)")

        # 3. Pattern Recognition
        if patterns:
            # Bull Flag
            if patterns.get('bull_flag', {}).get('is_flag'):
                reasons.append(patterns['bull_flag']['explanation'])
            elif patterns.get('bull_flag', {}).get('flag_score', 0) > 0.2:
                reasons.append(patterns['bull_flag']['explanation'])

            # GEX Walls
            if patterns.get('gex_wall', {}).get('wall_protection_score', 0) > 0.3:
                reasons.append(patterns['gex_wall']['explanation'])

            # Reversal
            if patterns.get('reversal', {}).get('is_reversal'):
                reasons.append(patterns['reversal']['explanation'])

        # 4. AI Confidence
        nn_score = row.get('nn_score', 0)
        if nn_score > 75:
            reasons.append(f"High Hive Mind confidence ({nn_score:.0f}%)")
        elif nn_score > 60:
            reasons.append(f"Moderate AI confidence ({nn_score:.0f}%)")
        elif nn_score < 40:
            warnings.append(f"Low AI confidence ({nn_score:.0f}%)")

        # 5. Technical Context
        rsi = row.get('rsi', 50)
        trend_score = row.get('trend_score_val', 0) if 'trend_score_val' in row else row.get('trend_score', 0)

        if rsi < 35:
            reasons.append(f"Oversold (RSI {rsi:.0f})")
        elif rsi > 70:
            warnings.append(f"Overbought (RSI {rsi:.0f})")

        # 6. Sector Context
        sector_status = row.get('sector_status', 'Unknown')
        if sector_status == 'Leading Sector':
            reasons.append("In leading sector")
        elif sector_status == 'Lagging Sector':
            if patterns and patterns.get('reversal', {}).get('is_reversal'):
                reasons.append("Lagging sector reversal play")
            else:
                warnings.append("In lagging sector")

        # 7. Quality
        quality = row.get('quality', 'Unknown')
        if quality == 'Quality Leader':
            reasons.append("Large cap quality")
        elif quality == 'Speculative':
            warnings.append("Small cap/speculative")

        # Build final explanation
        explanation_parts = []
        if reasons:
            explanation_parts.append(" | ".join(reasons[:5]))  # Limit to top 5 reasons
        if warnings:
            explanation_parts.append("⚠️ " + ", ".join(warnings[:3]))  # Limit warnings

        return " || ".join(explanation_parts) if explanation_parts else "Standard momentum signal"

    def apply_sector_capping(self, df, max_per_sector=None):
        """
        Apply sector capping to prevent over-concentration in any single sector.
        Returns filtered dataframe with max N picks per sector.
        """
        if max_per_sector is None:
            max_per_sector = MAX_PICKS_PER_SECTOR

        if df.empty or 'ticker' not in df.columns:
            return df

        # Get sector for each ticker
        def get_sector(ticker):
            return self.sector_map_local.get(ticker, 'Unknown')

        df = df.copy()
        df['_sector'] = df['ticker'].apply(get_sector)

        # Group by sector and take top N from each
        capped_dfs = []
        sector_counts = {}

        for _, row in df.iterrows():
            sector = row['_sector']
            sector_counts[sector] = sector_counts.get(sector, 0)

            if sector_counts[sector] < max_per_sector:
                capped_dfs.append(row)
                sector_counts[sector] += 1

        if capped_dfs:
            result = pd.DataFrame(capped_dfs)
            result = result.drop(columns=['_sector'], errors='ignore')

            # Log sector distribution
            print(f"  [RISK] Sector capping applied: {dict(sector_counts)}")
            return result

        return df.drop(columns=['_sector'], errors='ignore')

    # --- RESTORED PROCESS FLOW DATA ---
    def process_flow_data(self, file_map):
        print("\n[1/4] Processing Options & Dark Pool Data...")
        df_dp = self.safe_read(file_map.get('dp'), "Dark Pools")
        dp_stats = pd.DataFrame()
        if not df_dp.empty:
            df_dp['ticker'] = df_dp['ticker'].apply(self.normalize_ticker)
            if 'ext_hour_sold_codes' in df_dp.columns:
                ghost_codes = ['extended_hours_trade_late_or_out_of_sequence', 'sold_out_of_sequence']
                is_ghost = df_dp['ext_hour_sold_codes'].isin(ghost_codes)
                ghost_prints = df_dp[is_ghost & (df_dp['premium'] > 500_000)]  # Lowered from 1M to 500K
                if not ghost_prints.empty:
                    print(f"  [SHIELD] Detected {len(ghost_prints)} Signature Prints.")
                    for t, group in ghost_prints.groupby('ticker'): self.dp_support_levels[t] = group['price'].unique().tolist()
            if {'nbbo_ask', 'nbbo_bid'}.issubset(df_dp.columns):
                df_dp['sentiment'] = np.where(df_dp['price'] >= df_dp['nbbo_ask'], 1, np.where(df_dp['price'] <= df_dp['nbbo_bid'], -1, 0))
            else: df_dp['sentiment'] = 0
            df_dp['est_vol'] = df_dp['premium'] / (df_dp['price'] + 1e-9)
            dp_stats = df_dp.groupby('ticker').agg({'premium': 'sum', 'sentiment': 'mean', 'est_vol': 'sum'}).rename(columns={'premium': 'dp_total', 'sentiment': 'dp_sentiment'})
            dp_stats['dp_vwap'] = dp_stats['dp_total'] / (dp_stats['est_vol'] + 1e-9)
            dp_stats.drop(columns=['est_vol'], inplace=True)

        df_hot = self.safe_read(file_map.get('hot'), "Hot Chains")
        flow_stats = pd.DataFrame()
        if not df_hot.empty:
            df_hot['ticker'] = df_hot['option_symbol'].str.extract(r'([A-Z]+)').iloc[:, 0].apply(self.normalize_ticker)
            if 'next_earnings_date' in df_hot.columns:
                print("  [SPEED] Caching Earnings Dates...")
                valid_dates = df_hot.dropna(subset=['next_earnings_date'])
                self.earnings_map = dict(zip(valid_dates['ticker'], valid_dates['next_earnings_date']))
            df_hot['is_call'] = df_hot['option_symbol'].str.contains('C', regex=True)
            df_hot['net_prem'] = df_hot['premium'] * np.where(df_hot['is_call'], 1, -1)
            flow_stats = df_hot.groupby('ticker').agg({'premium': 'sum', 'net_prem': 'sum', 'iv': 'mean'}).rename(columns={'premium': 'opt_vol', 'net_prem': 'net_flow', 'iv': 'avg_iv'})

        df_oi = self.safe_read(file_map.get('oi'), "OI Changes")
        oi_stats = pd.DataFrame()
        if not df_oi.empty:
            if 'underlying_symbol' in df_oi.columns: df_oi['ticker'] = df_oi['underlying_symbol']
            df_oi['ticker'] = df_oi['ticker'].apply(self.normalize_ticker)
            oi_stats = df_oi.groupby('ticker').agg({'oi_change': 'sum'}).rename(columns={'oi_change': 'oi_change'})

        df_screener = self.safe_read(file_map.get('bot_lite'), "Stock Screener")
        if not df_screener.empty and 'ticker' in df_screener.columns:
            df_screener['ticker'] = df_screener['ticker'].apply(self.normalize_ticker)
            if 'marketcap' in df_screener.columns: self.cap_map = dict(zip(df_screener['ticker'], df_screener['marketcap']))
            if 'sector' in df_screener.columns: self.sector_map_local.update(dict(zip(df_screener['ticker'], df_screener['sector'])))

        df_bot = pd.DataFrame()
        greeks_stats = pd.DataFrame()
        if os.path.exists(self.optimized_bot_file):
            print(f"  [+] Found Optimized Dataset: {os.path.basename(self.optimized_bot_file)}")
            df_bot = self.safe_read(self.optimized_bot_file, "Optimized Gamma Data")
        elif file_map.get('bot_big') and os.path.exists(file_map.get('bot_big')):
            df_bot = self.optimize_large_dataset(file_map.get('bot_big'), date_stamp=None)
        elif not df_screener.empty:
            df_bot = df_screener.copy()
            if 'net_call_premium' in df_bot.columns:
                df_bot['screener_flow'] = df_bot['net_call_premium'] - df_bot['net_put_premium']
                df_bot['net_gamma'] = df_bot['screener_flow'] / 100.0

        if not df_bot.empty:
            target_gamma = 'authentic_gamma' if 'authentic_gamma' in df_bot.columns else 'net_gamma'
            if target_gamma in df_bot.columns:
                 if 'sector' in df_bot.columns:
                     valid = df_bot.dropna(subset=['sector'])
                     self.sector_map_local.update(dict(zip(valid['ticker'], valid['sector'])))
                 agg = {target_gamma: 'sum', 'equity_type': 'first', 'adj_iv': 'mean'}
                 if 'open_interest' in df_bot.columns: agg['open_interest'] = 'sum'
                 if 'net_delta' in df_bot.columns: agg['net_delta'] = 'sum'
                 greeks_stats = df_bot.groupby('ticker').agg(agg).rename(columns={target_gamma: 'net_gamma'})

        dfs = [d for d in [dp_stats, flow_stats, oi_stats, greeks_stats] if not d.empty]
        if not dfs: return pd.DataFrame()
        full_df = dfs[0]
        for d in dfs[1:]: full_df = pd.merge(full_df, d, how='outer', left_index=True, right_index=True)
        self.full_df = full_df.reset_index().rename(columns={'index': 'ticker'}).fillna(0)

        self.get_market_regime()
        self.full_df = self.generate_temporal_features(self.full_df)
        self.train_run_transformer()
        return self.full_df

    def calculate_technicals(self, history_df):
        if len(history_df) < 50: return None
        close = history_df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        trend_score = (close - sma20) / (sma20 + 1e-9)
        volatility = close.pct_change().rolling(20).std()
        dist_sma50 = (close.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
        div_score = 0.0
        if len(close) > 15:
            y_p = close.iloc[-10:].values
            y_r = rsi.iloc[-10:].values
            if y_p[-1] > y_p[0] and y_r[-1] < y_r[0]: div_score = 1.0

        return {
            'rsi': float(rsi.iloc[-1]),
            'trend_score': float(trend_score.iloc[-1]),
            'volatility': float(volatility.iloc[-1]),
            'flag_score': 0.0,
            'divergence_score': float(div_score),
            'sma_alignment': 1 if sma20.iloc[-1] > sma50.iloc[-1] else 0,
            'lagged_return_5d': float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) if len(close) > 6 else 0.0,
            'current_price': float(close.iloc[-1]),
            'dist_sma50': float(dist_sma50)
        }

    def enrich_market_data(self, flow_df):
        print("\n[2/4] Enriching with Price History (Deep Mode)...")
        if flow_df.empty: return self.full_df
        self.fetch_sector_history()
        cached_data = {}
        if os.path.exists(self.price_cache_file):
            try:
                df_cache = pd.read_csv(self.price_cache_file)
                cached_data = df_cache.set_index('ticker').to_dict('index')
            except: pass

        if 'ticker' not in flow_df.columns: return flow_df
        tickers = flow_df['ticker'].unique().tolist()
        to_fetch = [t for t in tickers if t not in cached_data and len(str(t)) < 8 and str(t) != 'nan']

        if to_fetch:
            print(f"  [FETCH] Downloading {len(to_fetch)} tickers...")
            batch_size = 50
            for i in range(0, len(to_fetch), batch_size):
                batch = to_fetch[i:i+batch_size]
                try:
                    data = yf.download(batch, period="3mo", group_by='ticker', progress=False, threads=True)
                    for t in batch:
                        try:
                            hist = data[t] if len(batch) > 1 else data
                            if not hist.empty:
                                metrics = self.calculate_technicals(hist)
                                if metrics: cached_data[t] = metrics
                        except: pass
                except: pass
            try:
                pd.DataFrame.from_dict(cached_data, orient='index').reset_index().rename(columns={'index':'ticker'}).to_csv(self.price_cache_file, index=False)
            except: pass

        self.market_breadth = self.calculate_market_breadth(cached_data)
        tech_df = pd.DataFrame.from_dict(cached_data, orient='index').reset_index().rename(columns={'index':'ticker'})
        if tech_df.empty:
            self.full_df = flow_df.copy()
            for c in ['trend_score', 'rsi']: self.full_df[c] = 0
            return self.full_df
        final_df = pd.merge(flow_df, tech_df, on='ticker', how='left')
        self.full_df = final_df[final_df['rsi'].notna()].fillna(0)
        return self.full_df

    def train_model(self, force_retrain=False):
        if self.full_df.empty: return
        print("\n[3/4] Training CatBoost (Left Brain)...")
        if 'lagged_return_5d' not in self.full_df.columns: self.full_df['lagged_return_5d'] = 0.0
        self.full_df['target'] = (self.full_df['lagged_return_5d'] > 0.02).astype(int)

        if self.full_df['target'].nunique() < 2: return

        tech_feats = ['rsi', 'trend_score', 'volatility', 'sma_alignment', 'divergence_score', 'dist_sma50']
        flow_feats = ['dp_sentiment', 'net_flow', 'avg_iv', 'net_gamma', 'oi_change', 'dp_total']
        temporal_feats = ['gamma_velocity', 'oi_accel']
        neural_feats = ['nn_score']

        all_possible = tech_feats + flow_feats + temporal_feats + neural_feats
        self.features_list = [f for f in all_possible if f in self.full_df.columns]

        X = self.full_df[self.features_list]
        y = self.full_df['target']
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        print("  [CATBOOST] Tuning Hyperparameters...")
        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'logging_level': 'Silent',
                'thread_count': -1
            }
            model = CatBoostClassifier(**param)
            return cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc').mean()

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30)
            cb = CatBoostClassifier(**study.best_params, logging_level='Silent')
            self.model = CalibratedClassifierCV(cb, method='sigmoid', cv=5)
            self.model.fit(X_scaled, y)
            self.model_trained = True
            print("  [SAVE] Saving CatBoost model to disk...")
        except Exception as e:
            print(f"  [!] Training failed: {e}")
            self.model_trained = False

    def predict(self):
        if self.full_df.empty: return None
        print("\n[4/4] Generating Predictions with Pattern Intelligence...")
        df = self.full_df.copy()

        # --- BASE SCORE FROM ML MODEL ---
        if self.model_trained:
            X = self.full_df[self.features_list]
            X_clean = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_clean)
            probs = self.model.predict_proba(X_scaled)[:, 1]
            df['raw_score'] = probs
        else:
            df['raw_score'] = 0.5

        df['trend_score_val'] = df['raw_score'] * 85
        df['ambush_score_val'] = df['raw_score'] * 80

        # --- FLOW SUPPORT CHECK ---
        flow_support = pd.Series(False, index=df.index)
        if 'net_gamma' in df.columns:
            flow_support |= (df['net_gamma'].abs() > 50)
        if 'dp_total' in df.columns:
            flow_support |= (df['dp_total'] > 1_000_000)
        df.loc[~flow_support, 'trend_score_val'] -= 40
        df.loc[~flow_support, 'ambush_score_val'] -= 40

        # --- NEURAL NETWORK BOOST ---
        if 'nn_score' in df.columns:
            boost = (df['nn_score'] - 50) * 0.5
            df['trend_score_val'] += boost
            df['ambush_score_val'] += boost

        # --- GAMMA VELOCITY BOOST ---
        if 'gamma_velocity' in df.columns:
            df.loc[df['gamma_velocity'] > 50, 'trend_score_val'] += 5

        # --- MACRO ADJUSTMENT (Phase 9) ---
        macro_adj = self.macro_data.get('adjustment', 0)
        if macro_adj != 0:
            print(f"  [MACRO] Applying {macro_adj:+.1f} point adjustment to all scores")
            df['trend_score_val'] += macro_adj
            df['ambush_score_val'] += macro_adj

        # --- TITAN EXECUTION LAYER (ATR STOP/PROFIT) ---
        price_db = self.history_mgr.db.get_price_df()
        price_lookup = price_db.set_index(['ticker', 'date'])['atr'].to_dict() if not price_db.empty else {}

        # Fallback ATR if not in DB
        if 'volatility' in df.columns and 'current_price' in df.columns:
            df['atr'] = df['volatility'] * df['current_price'] * 2  # Crude approx
        else:
            df['atr'] = 0

        if 'current_price' in df.columns:
            df['stop_loss'] = df['current_price'] - (2.5 * df['atr'])
            df['take_profit'] = df['current_price'] + (4.0 * df['atr'])
        else:
            df['stop_loss'] = 0
            df['take_profit'] = 0

        # --- INITIAL RANKING ---
        df['max_score'] = df[['trend_score_val', 'ambush_score_val']].max(axis=1)
        df = df.sort_values('max_score', ascending=False)
        top_candidates = df.head(75).copy()

        print(f"  [INFO] Base scoring complete. Running pattern detection on top 75 candidates...")

        # --- PHASE 9: PATTERN DETECTION & EXPLANATION GENERATION ---
        pattern_results = {}
        tickers_to_analyze = top_candidates['ticker'].tolist()

        # Batch fetch price history for pattern detection
        print(f"  [PATTERNS] Fetching price history for pattern analysis...")
        price_data = {}
        batch_size = 50
        for i in range(0, len(tickers_to_analyze), batch_size):
            batch = tickers_to_analyze[i:i+batch_size]
            try:
                data = yf.download(batch, period="3mo", group_by='ticker', progress=False, threads=True)
                for t in batch:
                    try:
                        hist = data[t] if len(batch) > 1 else data
                        if not hist.empty:
                            price_data[t] = hist
                    except:
                        pass
            except:
                pass

        # Run pattern detection on each candidate
        print(f"  [PATTERNS] Analyzing {len(tickers_to_analyze)} tickers for bull flags, GEX walls, and reversals...")
        for idx, row in top_candidates.iterrows():
            ticker = row['ticker']
            current_price = row.get('current_price', 0)
            eq_type = row.get('equity_type', 'Unknown')

            # Get price history for this ticker
            hist_df = price_data.get(ticker)

            # Run pattern detection
            patterns = {
                'bull_flag': self.detect_bull_flag(ticker, hist_df),
                'gex_wall': self.find_gex_walls(ticker, current_price),
                'reversal': self.detect_downtrend_reversal(ticker, hist_df)
            }
            pattern_results[ticker] = patterns

            # --- PATTERN-BASED SCORE ADJUSTMENTS ---

            # Bull flag bonus
            flag_score = patterns['bull_flag'].get('flag_score', 0)
            if flag_score > 0:
                bonus = flag_score * 10  # Up to 10 point bonus
                top_candidates.at[idx, 'trend_score_val'] = row['trend_score_val'] + bonus

            # GEX wall protection bonus
            wall_score = patterns['gex_wall'].get('wall_protection_score', 0)
            if wall_score > 0:
                bonus = wall_score * 8  # Up to 8 point bonus
                top_candidates.at[idx, 'trend_score_val'] = row['trend_score_val'] + bonus
                top_candidates.at[idx, 'ambush_score_val'] = row['ambush_score_val'] + bonus

            # Reversal setup bonus (for ambush strategy)
            reversal_score = patterns['reversal'].get('reversal_score', 0)
            if reversal_score > 0:
                bonus = reversal_score * 12  # Up to 12 point bonus for ambush
                top_candidates.at[idx, 'ambush_score_val'] = row['ambush_score_val'] + bonus

            # Store pattern flags
            top_candidates.at[idx, 'has_bull_flag'] = patterns['bull_flag'].get('is_flag', False)
            top_candidates.at[idx, 'has_gex_support'] = wall_score > 0.3
            top_candidates.at[idx, 'is_reversal_setup'] = patterns['reversal'].get('is_reversal', False)

            # Fundamental & Sector Analysis
            ctx = self.analyze_fundamentals_and_sector(ticker, eq_type)
            top_candidates.at[idx, 'quality'] = ctx['quality_label']
            top_candidates.at[idx, 'sector_status'] = ctx['sector_status']

            # DP Support Level
            if ticker in self.dp_support_levels:
                levels = [p for p in self.dp_support_levels[ticker] if p < current_price]
                if levels:
                    top_candidates.at[idx, 'dp_support'] = f"${max(levels):.2f}"
                else:
                    top_candidates.at[idx, 'dp_support'] = "None"
            else:
                top_candidates.at[idx, 'dp_support'] = "None"

            # GEX Wall Levels
            if patterns['gex_wall'].get('support_wall'):
                top_candidates.at[idx, 'gex_support'] = f"${patterns['gex_wall']['support_wall']:.2f}"
            else:
                top_candidates.at[idx, 'gex_support'] = "None"

        # --- GENERATE EXPLANATIONS ---
        print(f"  [EXPLAIN] Generating human-readable explanations...")
        for idx, row in top_candidates.iterrows():
            ticker = row['ticker']
            patterns = pattern_results.get(ticker, {})
            explanation = self.generate_signal_explanation(row.to_dict(), patterns)
            top_candidates.at[idx, 'explanation'] = explanation

        # --- SPLIT ETF AND STOCKS ---
        etf_candidates = top_candidates[top_candidates['quality'] == 'ETF'].sort_values('trend_score_val', ascending=False)
        stock_candidates = top_candidates[top_candidates['quality'] != 'ETF'].copy()

        # --- APPLY SECTOR CAPPING (Phase 9) ---
        print(f"\n  [RISK] Applying sector capping (max {MAX_PICKS_PER_SECTOR} per sector)...")
        stock_candidates = stock_candidates.sort_values('trend_score_val', ascending=False)
        stock_candidates = self.apply_sector_capping(stock_candidates)

        # --- FINAL SCORE FORMATTING ---
        stock_candidates['trend_score'] = stock_candidates['trend_score_val'].clip(0, 99.9).round(1)
        stock_candidates['ambush_score'] = stock_candidates['ambush_score_val'].clip(0, 99.9).round(1)
        if 'dist_sma50' in stock_candidates.columns:
            stock_candidates['ext'] = (stock_candidates['dist_sma50'] * 100).round(1)
        else:
            stock_candidates['ext'] = 0.0

        # Ensure all expected columns exist
        for col in ['gamma_velocity', 'nn_score', 'stop_loss', 'take_profit', 'explanation', 'gex_support']:
            if col not in stock_candidates.columns:
                stock_candidates[col] = 0.0 if col != 'explanation' else 'Standard signal'
            if col not in etf_candidates.columns:
                etf_candidates[col] = 0.0 if col != 'explanation' else 'Standard signal'

        # Pattern flags formatting
        for col in ['has_bull_flag', 'has_gex_support', 'is_reversal_setup']:
            if col not in stock_candidates.columns:
                stock_candidates[col] = False
            if col not in etf_candidates.columns:
                etf_candidates[col] = False

        # --- SUMMARY STATISTICS ---
        bull_flags = stock_candidates['has_bull_flag'].sum() if 'has_bull_flag' in stock_candidates.columns else 0
        gex_protected = stock_candidates['has_gex_support'].sum() if 'has_gex_support' in stock_candidates.columns else 0
        reversal_setups = stock_candidates['is_reversal_setup'].sum() if 'is_reversal_setup' in stock_candidates.columns else 0

        print(f"\n  [PATTERNS SUMMARY]")
        print(f"    Bull Flags Detected: {bull_flags}")
        print(f"    GEX Protected Positions: {gex_protected}")
        print(f"    Reversal Setups: {reversal_setups}")

        return stock_candidates, etf_candidates

if __name__ == "__main__":
    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING ENGINE RUN ---")

    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("  [INIT] Mounting Google Drive...")
            drive.mount('/content/drive')
    except ImportError: pass

    script_dir = os.getcwd()
    if '__file__' in globals(): script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = ["/content/drive/My Drive/colab", script_dir, os.getcwd(), "/content/drive/MyDrive", "/content/drive/MyDrive/Colab Notebooks", "/content"]
    def find_file(name):
        for p in search_paths:
            full = os.path.join(p, name)
            if os.path.exists(full): return full
        return None

    engine = SwingTradingEngine()
    data_dir = "/content/drive/My Drive/colab"
    if not os.path.exists(data_dir): data_dir = os.getcwd()
    engine.history_mgr.sync_history(engine, data_dir)

    today_str = datetime.now().strftime('%Y-%m-%d')
    files = {
        'dp': find_file(f"dp-eod-report-{today_str}.csv") or find_file("dp-eod-report.csv"),
        'hot': find_file(f"hot-chains-{today_str}.csv") or find_file("hot-chains.csv"),
        'oi': find_file(f"chain-oi-changes-{today_str}.csv") or find_file("chain-oi-changes.csv"),
        'bot_lite': find_file(f"stock-screener-{today_str}.csv") or find_file("stock-screener.csv"),
        'bot_big': find_file(f"bot-eod-report-{today_str}.csv") or find_file("bot-eod-report.csv")
    }

    if not files['bot_big']:
        files['bot_big'] = find_file("bot-eod-report-2025-12-09.csv")
        files['dp'] = find_file("dp-eod-report-2025-12-09.csv")
        files['hot'] = find_file("hot-chains-2025-12-09.csv")
        files['oi'] = find_file("chain-oi-changes-2025-12-09.csv")
        files['bot_lite'] = find_file("stock-screener-2025-12-09.csv")

    if not engine.process_flow_data(files).empty:
        engine.enrich_market_data(engine.full_df)
        engine.train_model()
        results = engine.predict()
        if results is not None:
            stocks_df, etfs_df = results

            # --- MACRO CONTEXT HEADER ---
            print("\n" + "="*80)
            print(f"GRANDMASTER ENGINE v9.0 - {datetime.now().strftime('%Y-%m-%d')}")
            print("="*80)
            print(f"MACRO REGIME: {engine.market_regime}")
            if hasattr(engine, 'macro_data') and engine.macro_data:
                m = engine.macro_data
                print(f"  VIX: {m.get('vix', 'N/A'):.2f} | TNX: {m.get('tnx', 'N/A'):.2f}% | DXY: {m.get('dxy', 'N/A'):.2f}")
                print(f"  Score Adjustment: {m.get('adjustment', 0):+.1f} points")

            # --- STRATEGY 1: MOMENTUM LEADERS ---
            trend_picks = stocks_df[stocks_df['trend_score'] > 80].sort_values('trend_score', ascending=False)
            print("\n" + "="*80)
            print(f"STRATEGY 1: MOMENTUM LEADERS (Trend Following)")
            print(f"Logic: CatBoost + Hive Mind + Bull Flag Detection + GEX Protection")
            print("="*80)

            # Display table
            display_cols = ['ticker', 'trend_score', 'quality', 'has_bull_flag', 'has_gex_support', 'nn_score', 'stop_loss', 'take_profit']
            display_cols = [c for c in display_cols if c in stocks_df.columns]
            if not trend_picks.empty:
                print(trend_picks[display_cols].head(8).to_string(index=False))

                # Show explanations for top picks
                print("\n  --- SIGNAL EXPLANATIONS ---")
                for _, row in trend_picks.head(5).iterrows():
                    explanation = row.get('explanation', 'N/A')
                    # Truncate long explanations for display
                    if len(str(explanation)) > 120:
                        explanation = str(explanation)[:117] + "..."
                    print(f"  {row['ticker']}: {explanation}")
            else:
                print("  [!] No candidates met strict criteria (>80).")

            # --- STRATEGY 2: AMBUSH PREDATORS ---
            # More lenient criteria for reversal plays
            ambush_picks = stocks_df[
                (stocks_df['ambush_score'] > 75) &
                ((stocks_df['is_reversal_setup'] == True) | (stocks_df['sector_status'] == 'Lagging Sector'))
            ].sort_values('ambush_score', ascending=False)

            print("\n" + "="*80)
            print(f"STRATEGY 2: AMBUSH PREDATORS (Counter-Trend Reversals)")
            print(f"Logic: Downtrend + DP Support + Divergence Detection")
            print("="*80)

            display_cols = ['ticker', 'ambush_score', 'quality', 'is_reversal_setup', 'dp_support', 'gex_support', 'stop_loss', 'take_profit']
            display_cols = [c for c in display_cols if c in stocks_df.columns]
            if not ambush_picks.empty:
                print(ambush_picks[display_cols].head(8).to_string(index=False))

                # Show explanations for ambush picks
                print("\n  --- SIGNAL EXPLANATIONS ---")
                for _, row in ambush_picks.head(5).iterrows():
                    explanation = row.get('explanation', 'N/A')
                    if len(str(explanation)) > 120:
                        explanation = str(explanation)[:117] + "..."
                    print(f"  {row['ticker']}: {explanation}")
            else:
                print("  [!] No reversal candidates found.")

            # --- STRATEGY 3: BULL FLAG SPECIAL ---
            flag_picks = stocks_df[stocks_df['has_bull_flag'] == True].sort_values('trend_score', ascending=False)
            if not flag_picks.empty:
                print("\n" + "="*80)
                print(f"STRATEGY 3: BULL FLAG BREAKOUTS")
                print(f"Logic: Strong pole + Tight consolidation + Declining volume")
                print("="*80)
                display_cols = ['ticker', 'trend_score', 'quality', 'nn_score', 'gex_support', 'stop_loss', 'take_profit']
                display_cols = [c for c in display_cols if c in stocks_df.columns]
                print(flag_picks[display_cols].head(5).to_string(index=False))

                print("\n  --- SIGNAL EXPLANATIONS ---")
                for _, row in flag_picks.head(3).iterrows():
                    explanation = row.get('explanation', 'N/A')
                    if len(str(explanation)) > 120:
                        explanation = str(explanation)[:117] + "..."
                    print(f"  {row['ticker']}: {explanation}")

            # --- STRATEGY 4: ETF SWING ---
            if not etfs_df.empty:
                print("\n" + "="*80)
                print(f"STRATEGY 4: ETF SWING")
                print("="*80)
                etf_display = ['ticker', 'trend_score_val', 'sector_status', 'net_gamma', 'nn_score', 'stop_loss', 'take_profit']
                etf_display = [c for c in etf_display if c in etfs_df.columns]
                print(etfs_df[etf_display].head(5).to_string(index=False))

            # --- SAVE OUTPUT ---
            out_path = "/content/drive/My Drive/colab/swing_signals_v9_grandmaster.csv" if COLAB_ENV else os.path.join(engine.base_dir, "swing_signals_v9_grandmaster.csv")
            try:
                final_output = pd.concat([stocks_df, etfs_df])
                final_output.to_csv(out_path, index=False)
                print(f"\n[SUCCESS] Saved comprehensive report with explanations: {out_path}")
            except Exception as e:
                out_path = os.path.join(engine.base_dir, "swing_signals_v9_grandmaster.csv")
                try:
                    pd.concat([stocks_df, etfs_df]).to_csv(out_path, index=False)
                    print(f"\n[FALLBACK] Saved locally to {out_path}")
                except:
                    print(f"\n[ERROR] Could not save output: {e}")

            # --- SUMMARY ---
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            device_name = "CPU"
            if torch.cuda.is_available():
                device_name = f"GPU ({torch.cuda.get_device_name(0)})"

            print("\n" + "="*80)
            print("RUN SUMMARY")
            print("="*80)
            print(f"  Duration: {int(mins)}m {int(secs)}s")
            print(f"  Device: {device_name}")
            print(f"  Total Candidates Analyzed: {len(stocks_df) + len(etfs_df)}")
            print(f"  Momentum Leaders (>80): {len(trend_picks)}")
            print(f"  Ambush Setups: {len(ambush_picks)}")
            print(f"  Bull Flags: {len(flag_picks) if not flag_picks.empty else 0}")

            msg = f"v9.0 | Duration: {int(mins)}m {int(secs)}s | Device: {device_name} | Items: {len(stocks_df) + len(etfs_df)}"
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- {msg} ---")

            # Log run history
            try:
                hist_path = "/content/drive/My Drive/colab/run_history.txt" if COLAB_ENV else os.path.join(engine.base_dir, "run_history.txt")
                with open(hist_path, "a") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
            except:
                pass

            # Persist DB
            engine.history_mgr.save_db()
    else:
        print("[CRITICAL] Missing data files. Please ensure Unusual Whales data is in the expected location.")