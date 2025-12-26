# -*- coding: utf-8 -*-
"""
Grandmaster Engine v11.5 - Data Loading Components

Database management and history synchronization:
- TitanDB: SQLite backbone for flow and price history
- HistoryManager: Ingests Unusual Whales files, syncs prices via Alpaca
"""

import os
import glob
import shutil
import random
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from .utils import (
    fetch_alpaca_batch,
    get_market_last_close_date,
    is_weekend,
)

# =============================================================================
# TITAN DATABASE MANAGER (SQLITE BACKBONE)
# =============================================================================
class TitanDB:
    """
    SQLite-backed database for flow and price history.

    Tables:
    - flow_history: Options flow data from Unusual Whales
    - price_history: OHLCV + ATR data from Alpaca
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def init_db(self):
        """Initialize database tables."""
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
        """Insert or update flow history records."""
        if df.empty:
            return

        cols = ['ticker', 'date', 'net_gamma', 'authentic_gamma',
                'net_delta', 'open_interest', 'adj_iv', 'equity_type']
        valid_df = df[cols].copy()
        valid_df['date'] = valid_df['date'].astype(str)

        # Bulk Insert/Replace
        valid_df.to_sql('temp_flow', self.conn, if_exists='replace', index=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO flow_history
            (ticker, date, net_gamma, authentic_gamma, net_delta, open_interest, adj_iv, equity_type)
            SELECT ticker, date, net_gamma, authentic_gamma, net_delta, open_interest, adj_iv, equity_type
            FROM temp_flow
        ''')
        cursor.execute('DROP TABLE temp_flow')
        self.conn.commit()

    def upsert_prices(self, df):
        """Insert or update price history records."""
        if df.empty:
            return

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
        """Get all flow history as DataFrame."""
        return pd.read_sql("SELECT * FROM flow_history", self.conn)

    def get_price_df(self):
        """Get all price history as DataFrame."""
        return pd.read_sql("SELECT * FROM price_history", self.conn)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# =============================================================================
# HISTORY MANAGER
# =============================================================================
class HistoryManager:
    """
    Manages historical data synchronization:
    - Ingests Unusual Whales flow files into TitanDB
    - Syncs price history via Alpaca API
    - Handles incremental updates and random refresh
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.local_db_path = os.path.abspath("titan.db")
        self.drive_db_path = os.path.join(base_dir, "titan.db")

        # Debug: Show paths being used
        print(f"  [TITAN] Base dir: {base_dir}")
        print(f"  [TITAN] Local DB: {self.local_db_path}")
        print(f"  [TITAN] Drive DB: {self.drive_db_path}")

        # Database Setup
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
        """Load list of already processed files."""
        if not os.path.exists(self.log_file):
            return set()
        with open(self.log_file, 'r') as f:
            return set(line.strip() for line in f)

    def _update_log(self, filename):
        """Add filename to processed log."""
        with open(self.log_file, 'a') as f:
            f.write(f"{filename}\n")
        self.processed_files.add(filename)

    def save_db(self):
        """Persist database to Drive."""
        self.db.close()
        try:
            if os.path.abspath(self.drive_db_path) != self.local_db_path:
                shutil.copy(self.local_db_path, self.drive_db_path)
                print(f"  [TITAN] Database persisted to Drive: {self.drive_db_path}")
            else:
                print(f"  [TITAN] Local and Drive paths are same - no copy needed.")
        except Exception as e:
            print(f"  [!] Failed to save DB to Drive: {e}")

    def sync_history(self, engine, data_folder):
        """
        Ingests Unusual Whales flow files into the DB.

        Args:
            engine: SwingTradingEngine instance (for optimize_large_dataset)
            data_folder: Path to folder containing bot-eod-report-*.csv files
        """
        # v11.2 OPTIMIZATION: Check if DB already has today's data
        try:
            existing_df = self.db.get_history_df()

            if existing_df.empty:
                print(f"  [HISTORY] DB is empty. Proceeding with file scan...")
            else:
                latest_date_str = existing_df['date'].max()
                print(f"  [HISTORY] DB latest date (raw): {latest_date_str}")

                latest_date = pd.to_datetime(latest_date_str)
                days_old = (datetime.now() - latest_date).days

                print(f"  [HISTORY] DB age: {days_old} days old")

                if days_old <= 1:
                    print(f"  [HISTORY] Flow DB current (latest: {latest_date.date()}). Skipping scan.")
                    self.sync_prices()
                    return
                else:
                    print(f"  [HISTORY] DB is stale ({days_old} days old). Scanning for new files...")

        except Exception as e:
            print(f"  [HISTORY] DB check failed: {str(e)[:80]}")
            print(f"  [HISTORY] Error type: {type(e).__name__}")

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
                except ValueError:
                    pass

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

        # Trigger price sync after flow ingest
        self.sync_prices()

    def sync_prices(self):
        """
        Alpaca-Powered Oracle.
        Syncs price history with incremental updates and random refresh.
        """
        print("\n[ORACLE v10.10] Syncing Price History via Alpaca...")

        # Check if DB is empty BEFORE checking for weekend
        price_df = self.db.get_price_df()
        is_empty = price_df.empty

        if is_weekend() and not is_empty:
            print("  [ORACLE] Weekend detected & DB exists - skipping sync.")
            return

        # 1. Identify what needs fetching
        try:
            hist_df = self.db.get_history_df()
            if hist_df.empty:
                return
            all_tickers = hist_df['ticker'].unique().tolist()
        except Exception as e:
            print(f"  [ORACLE] Failed to load history: {e}")
            return

        price_df = self.db.get_price_df()
        existing_tickers = set(price_df['ticker'].unique()) if not price_df.empty else set()

        tickers_need_full = []
        tickers_need_incremental = []
        market_last_close = get_market_last_close_date()

        # Logic: Sort tickers into 'New' or 'Stale'
        for t in all_tickers:
            if not isinstance(t, str):
                continue

            if t not in existing_tickers:
                tickers_need_full.append(t)
            else:
                # Check if stale (older than 3 days)
                try:
                    ticker_data = price_df[price_df['ticker'] == t]
                    if not ticker_data.empty:
                        last_date_str = ticker_data['date'].max()
                        last_date = pd.to_datetime(last_date_str).date()
                        days_stale = (market_last_close - last_date).days
                        if days_stale > 3:
                            tickers_need_incremental.append(t)
                except:
                    tickers_need_full.append(t)

        # Logic: Random 5% refresh for data quality
        refresh_candidates = [t for t in existing_tickers if t not in tickers_need_incremental]
        if refresh_candidates:
            refresh_batch = random.sample(
                list(refresh_candidates),
                int(len(refresh_candidates) * 0.05)
            )
            tickers_need_full.extend(refresh_batch)

        # Deduplicate
        tickers_need_full = list(set(tickers_need_full))

        # Combine all work into one big Alpaca batch
        to_fetch_list = list(set(tickers_need_full + tickers_need_incremental))

        if not to_fetch_list:
            print("  [ORACLE] Price DB is up to date.")
            return

        print(f"  [ORACLE] Fetching {len(to_fetch_list)} tickers (Full+Inc+Refresh)...")

        # 2. Execute Fetch
        start_date_2y = datetime.now() - timedelta(days=730)
        fetched_data = fetch_alpaca_batch(to_fetch_list, start_date=start_date_2y)

        # 3. Process & Save
        new_data = []
        successful_fetches = 0

        for ticker, hist in fetched_data.items():
            try:
                # ATR Calculation
                high = hist['High']
                low = hist['Low']
                close = hist['Close']
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().values

                # Format Rows
                for idx, (date_val, row) in enumerate(hist.iterrows()):
                    atr_val = atr[idx] if idx < len(atr) else 0
                    if pd.isna(atr_val):
                        atr_val = 0

                    new_data.append({
                        'ticker': ticker,
                        'date': str(date_val),
                        'close': float(row['Close']),
                        'atr': float(atr_val)
                    })
                successful_fetches += 1
            except:
                pass

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.db.upsert_prices(new_df)
            print(f"  [ORACLE] Success! Saved {len(new_df)} rows for {successful_fetches} tickers.")
        else:
            print("  [ORACLE] Warning: No valid data fetched.")


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    'TitanDB',
    'HistoryManager',
]
