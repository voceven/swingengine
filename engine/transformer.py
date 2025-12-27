# -*- coding: utf-8 -*-
"""Transformer/Hive Mind Training Functions for Swing Trading Engine v12

This module contains the neural network ensemble ("Hive Mind") that provides
the nn_score predictions used for momentum ranking.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

from .neural import SwingTransformer
from .config import PERFORMANCE_CONFIG


def prepare_supervised_data(history_mgr, nn_scaler, window_size=3, lookahead=1):
    """Prepare windowed sequences for transformer training.

    Returns:
        tuple: (X_data, y_data, inference_tickers, feature_cols) or (None, None, None, None)
    """
    print(f"\n[TRANSFORMER] Preparing Supervised Data (Window={window_size}, Lookahead={lookahead}d)...")
    hist_df = history_mgr.db.get_history_df()
    if hist_df.empty:
        return None, None, None, None

    needed_cols = ['ticker', 'date', 'net_gamma', 'net_delta', 'open_interest', 'adj_iv']
    valid_cols = [c for c in needed_cols if c in hist_df.columns]
    if len(valid_cols) < 4:
        return None, None, None, None

    price_df = history_mgr.db.get_price_df()
    if price_df.empty:
        return _prepare_inference_only(hist_df, valid_cols, nn_scaler, window_size)

    # v10.6.1: Robust date parsing - yfinance sometimes leaks header "Ticker" into data
    price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
    price_df = price_df.dropna(subset=['date'])
    if price_df.empty:
        return _prepare_inference_only(hist_df, valid_cols, nn_scaler, window_size)

    df = hist_df[valid_cols].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['ticker', 'date'])

    feature_cols = [c for c in valid_cols if c not in ['ticker', 'date']]
    df[feature_cols] = nn_scaler.fit_transform(df[feature_cols])

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

    if not sequences:
        return None, None, None, None

    X = np.array(sequences)
    y = np.array(targets)
    print(f"  [TRANSFORMER] Created {len(X)} sequences. Labeled Training Data: {count_labeled}.")
    return X, y, tickers_list, feature_cols


def _prepare_inference_only(hist_df, valid_cols, nn_scaler, window_size):
    """Fallback when no price data available - inference only mode."""
    df = hist_df[valid_cols].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    feature_cols = [c for c in valid_cols if c not in ['ticker', 'date']]
    df[feature_cols] = nn_scaler.fit_transform(df[feature_cols])

    sequences, targets, tickers_list = [], [], []
    for ticker, group in df.groupby('ticker'):
        if len(group) >= window_size:
            sequences.append(group[feature_cols].values[-window_size:])
            targets.append(-1)
            tickers_list.append(ticker)

    if not sequences:
        return None, None, None, None
    return np.array(sequences), np.array(targets), tickers_list, feature_cols


def train_hive_mind(history_mgr, nn_scaler, device, num_models=5):
    """Train ensemble of transformers (Hive Mind) and return nn_score predictions.

    Args:
        history_mgr: HistoryManager instance with database access
        nn_scaler: StandardScaler for feature normalization
        device: torch.device for computation
        num_models: Number of ensemble members (default 5)

    Returns:
        pd.DataFrame with columns ['ticker', 'nn_score'] or None
    """
    # Prepare data with fallback to smaller window
    X_data, y_data, inference_tickers, feature_cols = prepare_supervised_data(
        history_mgr, nn_scaler, window_size=3, lookahead=1
    )
    if X_data is None or np.sum(y_data != -1) == 0:
        X_data, y_data, inference_tickers, feature_cols = prepare_supervised_data(
            history_mgr, nn_scaler, window_size=2, lookahead=1
        )

    if X_data is None:
        return None

    train_mask = y_data != -1
    X_train = X_data[train_mask]
    y_train = y_data[train_mask]
    X_infer = X_data[~train_mask]

    input_size = len(feature_cols)
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
            epochs = PERFORMANCE_CONFIG.get('transformer_epochs', 30)

            for epoch in range(epochs):
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

        # GPU memory cleanup
        try:
            del X_tensor, y_tensor
            if X_infer_tensor is not None:
                del X_infer_tensor
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
    else:
        return None

    if ensemble_preds:
        avg_preds = np.mean(ensemble_preds, axis=0)
        nn_df = pd.DataFrame({'ticker': inference_tickers, 'nn_score': avg_preds})

        # Normalize to 0-100 scale
        min_s, max_s = nn_df['nn_score'].min(), nn_df['nn_score'].max()
        if max_s - min_s > 0.0001:
            nn_df['nn_score'] = (nn_df['nn_score'] - min_s) / (max_s - min_s) * 100
        else:
            nn_df['nn_score'] = nn_df['nn_score'] * 100

        return nn_df

    return None
