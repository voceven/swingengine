# -*- coding: utf-8 -*-
"""
Grandmaster Engine v12 - Data Preparation

Triple-barrier labeling and sequence preparation for ML training.
"""

import numpy as np
import pandas as pd

__all__ = [
    'triple_barrier_labels',
    'prepare_tcn_sequences',
]


def triple_barrier_labels(price_df, tickers, holding_period=5, pt_multiplier=1.5,
                          sl_multiplier=1.0, vol_lookback=20):
    """
    Generate triple-barrier labels for realistic trade outcome prediction.

    Instead of: "Did price go up 2% in 5 days?" (ignores stop-loss)
    Now: "Would a trade with ATR-based TP/SL have won?"

    Args:
        price_df: DataFrame with columns [ticker, date, close, atr]
        tickers: List of tickers to label
        holding_period: Max days to hold (vertical barrier)
        pt_multiplier: Take-profit = ATR * multiplier
        sl_multiplier: Stop-loss = ATR * multiplier
        vol_lookback: Days for ATR calculation

    Returns:
        DataFrame with [ticker, date, label, barrier_hit]
        label: 1 (TP hit), -1 (SL hit), 0 (time expired neutral)
    """
    results = []

    price_df = price_df.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])

    has_atr = 'atr' in price_df.columns

    for ticker in tickers:
        ticker_data = price_df[price_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        if len(ticker_data) < vol_lookback + holding_period + 1:
            continue

        # Use pre-calculated ATR if available
        if has_atr and ticker_data['atr'].notna().any():
            ticker_data['atr_use'] = ticker_data['atr']
        else:
            ticker_data['atr_use'] = ticker_data['close'].pct_change().abs().rolling(vol_lookback).mean() * ticker_data['close']

        for i in range(vol_lookback, len(ticker_data) - holding_period):
            entry_date = ticker_data.loc[i, 'date']
            entry_price = ticker_data.loc[i, 'close']
            atr = ticker_data.loc[i, 'atr_use']

            if pd.isna(atr) or atr <= 0 or pd.isna(entry_price):
                continue

            take_profit = entry_price + (atr * pt_multiplier)
            stop_loss = entry_price - (atr * sl_multiplier)

            future_slice = ticker_data.loc[i+1:i+holding_period]

            tp_hit = None
            sl_hit = None

            for j, row in future_slice.iterrows():
                close_price = row['close']
                if pd.isna(close_price):
                    continue
                if close_price >= take_profit and tp_hit is None:
                    tp_hit = j - i
                if close_price <= stop_loss and sl_hit is None:
                    sl_hit = j - i

            if tp_hit is not None and (sl_hit is None or tp_hit <= sl_hit):
                label = 1
                barrier_hit = 'TP'
            elif sl_hit is not None:
                label = -1
                barrier_hit = 'SL'
            else:
                final_price = future_slice['close'].iloc[-1] if len(future_slice) > 0 else entry_price
                final_return = (final_price - entry_price) / entry_price if entry_price > 0 else 0
                label = 1 if final_return > 0.01 else (-1 if final_return < -0.01 else 0)
                barrier_hit = 'TIME'

            results.append({
                'ticker': ticker,
                'date': entry_date,
                'label': label,
                'barrier_hit': barrier_hit,
                'entry_price': entry_price,
                'atr': atr
            })

    return pd.DataFrame(results)


def prepare_tcn_sequences(price_df, feature_df, tickers, seq_len=10,
                          feature_cols=None, max_sequences=50000):
    """
    Prepare real sequential data for TCN training.

    Creates actual temporal sequences showing how features evolved over time.

    Args:
        price_df: DataFrame with [ticker, date, close, atr]
        feature_df: DataFrame with [ticker, ...features...]
        tickers: List of tickers
        seq_len: Sequence length (days of history per sample)
        feature_cols: List of feature columns to use
        max_sequences: Maximum sequences to return (memory limit)

    Returns:
        X_seq: np.array of shape (n_samples, seq_len, n_features)
        y_seq: np.array of labels
        ticker_list: List of tickers for each sample
    """
    sequences = []
    labels = []
    ticker_list = []

    price_df = price_df.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])

    for ticker in tickers:
        ticker_prices = price_df[price_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        if len(ticker_prices) < seq_len + 6:
            continue

        # Calculate features from close price
        ticker_prices['returns'] = ticker_prices['close'].pct_change()
        ticker_prices['volatility'] = ticker_prices['returns'].rolling(10).std()

        # RSI
        delta = ticker_prices['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        ticker_prices['rsi'] = 100 - (100 / (1 + rs))

        # Momentum
        ticker_prices['momentum_5d'] = ticker_prices['close'].pct_change(5)
        ticker_prices['momentum_10d'] = ticker_prices['close'].pct_change(10)

        # ATR normalized
        if 'atr' in ticker_prices.columns and ticker_prices['atr'].notna().any():
            ticker_prices['atr_norm'] = ticker_prices['atr'] / ticker_prices['close']
        else:
            ticker_prices['atr_norm'] = ticker_prices['volatility'] * np.sqrt(10)

        tcn_features = ['returns', 'volatility', 'rsi', 'momentum_5d', 'momentum_10d', 'atr_norm']

        # Normalize
        for col in tcn_features:
            if col in ticker_prices.columns:
                mean = ticker_prices[col].mean()
                std = ticker_prices[col].std() + 1e-10
                ticker_prices[col] = (ticker_prices[col] - mean) / std

        # Create sequences
        for i in range(20, len(ticker_prices) - seq_len - 5):
            seq_data = ticker_prices.loc[i:i+seq_len-1, tcn_features].values

            if seq_data.shape[0] != seq_len or np.isnan(seq_data).any():
                continue

            future_price = ticker_prices.loc[i+seq_len+4, 'close'] if i+seq_len+4 < len(ticker_prices) else None
            current_price = ticker_prices.loc[i+seq_len-1, 'close']

            if future_price is None or current_price <= 0:
                continue

            forward_return = (future_price - current_price) / current_price
            label = 1 if forward_return > 0.02 else 0

            sequences.append(seq_data)
            labels.append(label)
            ticker_list.append(ticker)

    if not sequences:
        return None, None, None

    if len(sequences) > max_sequences:
        print(f"  [TCN] Sampling {max_sequences} from {len(sequences)} sequences (memory limit)")
        indices = np.random.choice(len(sequences), max_sequences, replace=False)
        sequences = [sequences[i] for i in indices]
        labels = [labels[i] for i in indices]
        ticker_list = [ticker_list[i] for i in indices]

    return np.array(sequences), np.array(labels), ticker_list
