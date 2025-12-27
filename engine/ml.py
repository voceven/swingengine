# -*- coding: utf-8 -*-
"""
engine/ml.py - Machine Learning Training Functions for Swing Trading Engine v12

Extracted ML training functions that take required data as parameters.
This enables modular testing and reduces the main engine file size.

Functions:
- train_catboost: Train CatBoost classifier with Optuna tuning
- train_tabnet: Train TabNet attention-based classifier
- train_tcn: Train Temporal Convolutional Network
- train_elasticnet: Train ElasticNet linear baseline
- train_meta_learner: Train meta-learner on base model predictions
- train_ensemble: Full ensemble training pipeline
"""

import numpy as np
import time
import os
import json
import gc

# ML imports
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import torch
import torch.nn as nn

from .config import PERFORMANCE_CONFIG
from .neural import TCN
from .data_prep import triple_barrier_labels, prepare_tcn_sequences


def train_catboost(X_scaled, y, has_gpu=False, n_trials=25, max_iterations=500, max_depth=10, cv_folds=5):
    """
    Train CatBoost classifier with Optuna hyperparameter tuning.

    Args:
        X_scaled: Scaled feature matrix
        y: Target labels
        has_gpu: Whether GPU is available
        n_trials: Number of Optuna trials
        max_iterations: Maximum CatBoost iterations
        max_depth: Maximum tree depth
        cv_folds: Cross-validation folds

    Returns:
        tuple: (model, best_params, auc_score) or (None, None, 0.0) on failure
    """
    if not CATBOOST_AVAILABLE or not OPTUNA_AVAILABLE:
        print("  [!] CatBoost or Optuna not available")
        return None, None, 0.0

    print(f"  [CATBOOST] Tuning with {'GPU' if has_gpu else 'CPU'}...")

    def catboost_objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 150, max_iterations),
            'depth': trial.suggest_int('depth', 5, max_depth),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'task_type': 'GPU' if has_gpu else 'CPU',
            'devices': '0' if has_gpu else None,
            'early_stopping_rounds': 50,
            'logging_level': 'Silent'
        }
        if not has_gpu:
            param['thread_count'] = -1
        model = CatBoostClassifier(**param)
        return cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='roc_auc').mean()

    try:
        study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study_cb.optimize(catboost_objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study_cb.best_params.copy()
        best_params['task_type'] = 'GPU' if has_gpu else 'CPU'
        if has_gpu:
            best_params['devices'] = '0'
        else:
            best_params['thread_count'] = -1
        best_params['early_stopping_rounds'] = 50
        best_params['logging_level'] = 'Silent'

        model = CatBoostClassifier(**best_params)
        model.fit(X_scaled, y, verbose=False)
        auc = study_cb.best_value
        print(f"  [CATBOOST] AUC: {auc:.4f} (iter={best_params.get('iterations')}, depth={best_params.get('depth')})")
        return model, best_params, auc
    except Exception as e:
        print(f"  [!] CatBoost training failed: {e}")
        return None, None, 0.0


def train_tabnet(X_scaled, y, has_gpu=False):
    """
    Train TabNet attention-based classifier.

    Args:
        X_scaled: Scaled feature matrix
        y: Target labels
        has_gpu: Whether GPU is available

    Returns:
        tuple: (model, auc_score) or (None, 0.0) on failure
    """
    if not TABNET_AVAILABLE:
        print(f"  [TABNET] Not available, skipping (install: pip install pytorch-tabnet)")
        return None, 0.0

    print(f"  [TABNET] Training attention-based model...")
    try:
        model = TabNetClassifier(
            n_d=16, n_a=16,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            device_name='cuda' if has_gpu else 'cpu',
            verbose=0
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=43, stratify=y
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=100,
            patience=15,
            batch_size=256
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        print(f"  [TABNET] AUC: {auc:.4f} (attention-based feature interactions)")
        return model, auc
    except Exception as e:
        print(f"  [!] TabNet training failed: {e}, using fallback")
        return None, 0.0


def train_tcn(price_df, full_df, tickers, seq_len=10, has_gpu=False, max_tickers=150):
    """
    Train Temporal Convolutional Network on real sequential data.

    Args:
        price_df: Price history DataFrame
        full_df: Full feature DataFrame
        tickers: List of tickers to train on
        seq_len: Sequence length for TCN
        has_gpu: Whether GPU is available
        max_tickers: Maximum tickers to sample (memory efficiency)

    Returns:
        tuple: (model, n_features, seq_len, auc_score) or (None, 6, 10, 0.0) on failure
    """
    print(f"  [TCN] Training temporal convolutional network with REAL sequences...")

    if price_df.empty:
        print("  [!] No price data available for TCN sequences")
        return None, 6, seq_len, 0.0

    if not tickers:
        print("  [!] No tickers available for TCN sequences")
        return None, 6, seq_len, 0.0

    try:
        device = torch.device('cuda' if has_gpu else 'cpu')

        # Sample tickers for memory efficiency
        if len(tickers) > max_tickers:
            import random
            random.seed(42)
            tickers = random.sample(tickers, max_tickers)
            print(f"  [TCN] Sampled {max_tickers} tickers for memory efficiency")

        # Generate real sequential data
        print(f"  [TCN] Building temporal sequences for {len(tickers)} tickers...")
        X_seq_np, y_seq_np, tcn_tickers = prepare_tcn_sequences(
            price_df, full_df, tickers, seq_len=seq_len
        )

        if X_seq_np is None or len(X_seq_np) < 100:
            raise ValueError(f"Insufficient sequence data: {len(X_seq_np) if X_seq_np is not None else 0} samples")

        n_features = X_seq_np.shape[2]
        print(f"  [TCN] Created {len(X_seq_np)} real sequences (shape: {X_seq_np.shape})")

        # Train/val split
        split_idx = int(len(X_seq_np) * 0.8)
        X_train_np, X_val_np = X_seq_np[:split_idx], X_seq_np[split_idx:]
        y_train_np, y_val_np = y_seq_np[:split_idx], y_seq_np[split_idx:]

        # Create DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_np),
            torch.FloatTensor(y_train_np).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        X_val_tcn = torch.FloatTensor(X_val_np)
        y_val_tcn = torch.FloatTensor(y_val_np).unsqueeze(1)

        # Initialize and train model
        model = TCN(
            input_size=n_features,
            num_channels=[32, 32, 16],
            kernel_size=3,
            dropout=0.2
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        best_val_loss, patience_counter = float('inf'), 0
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X.to(device))
                loss = criterion(outputs, batch_y.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_tcn.to(device))
                val_loss = criterion(val_out, y_val_tcn.to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        # Calculate AUC
        model.eval()
        with torch.no_grad():
            tcn_preds = model(X_val_tcn.to(device)).cpu().numpy().flatten()
        auc = roc_auc_score(y_val_tcn.numpy().flatten(), tcn_preds)
        print(f"  [TCN] AUC: {auc:.4f} (real temporal pattern detection)")
        return model, n_features, seq_len, auc
    except Exception as e:
        print(f"  [!] TCN training failed: {e}, using fallback")
        return None, 6, seq_len, 0.0


def train_elasticnet(X_scaled, y, cv_folds=5):
    """
    Train ElasticNet (L1+L2 regularized) linear baseline.

    Args:
        X_scaled: Scaled feature matrix
        y: Target labels
        cv_folds: Cross-validation folds

    Returns:
        tuple: (model, auc_score) or (None, 0.0) on failure
    """
    print(f"  [ELASTICNET] Training linear baseline...")
    try:
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=1000, random_state=44, n_jobs=-1
        )
        model.fit(X_scaled, y)
        preds = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='roc_auc')
        auc = preds.mean()
        print(f"  [ELASTICNET] AUC: {auc:.4f} (linear regularized baseline)")
        return model, auc
    except Exception as e:
        print(f"  [!] ElasticNet training failed: {e}")
        return None, 0.0


def train_meta_learner(X_scaled, y, catboost_model, tabnet_model=None, elasticnet_model=None, cv_folds=5):
    """
    Train meta-learner on base model predictions.

    Args:
        X_scaled: Scaled feature matrix
        y: Target labels
        catboost_model: Trained CatBoost model (required)
        tabnet_model: Trained TabNet model (optional)
        elasticnet_model: Trained ElasticNet model (optional)
        cv_folds: Cross-validation folds

    Returns:
        tuple: (meta_learner, model_names, ensemble_auc)
    """
    print(f"  [LEVEL-2] Training meta-learner with diverse ensemble...")

    # CatBoost predictions (always available)
    catboost_preds = cross_val_predict(catboost_model, X_scaled, y, cv=cv_folds, method='predict_proba')[:, 1]
    model_names, preds_list = ['CB'], [catboost_preds]

    # TabNet predictions
    if tabnet_model is not None:
        tabnet_preds = tabnet_model.predict_proba(X_scaled)[:, 1]
        preds_list.append(tabnet_preds)
        model_names.append('TN')

    # TCN skipped in meta-learner (uses different features)

    # ElasticNet predictions
    if elasticnet_model is not None:
        elasticnet_preds = elasticnet_model.predict_proba(X_scaled)[:, 1]
        preds_list.append(elasticnet_preds)
        model_names.append('EN')

    # Stack and train
    X_meta = np.column_stack(preds_list)
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_meta, y)

    # Calculate ensemble AUC
    meta_preds = meta_learner.predict_proba(X_meta)[:, 1]
    ensemble_auc = roc_auc_score(y, meta_preds)

    # Display weights
    weights_str = ', '.join([f"{name}={meta_learner.coef_[0][i]:.3f}" for i, name in enumerate(model_names)])
    print(f"  [META-LEARNER] Ensemble AUC: {ensemble_auc:.4f}")
    print(f"  [META-LEARNER] Weights: {weights_str}")

    return meta_learner, model_names, ensemble_auc


def train_ensemble(full_df, history_mgr, imputer, scaler, features_list,
                   cache_paths, market_regime, force_retrain=False):
    """
    Full ensemble training pipeline with caching.

    Args:
        full_df: Full feature DataFrame
        history_mgr: HistoryManager instance for price data
        imputer: SimpleImputer instance
        scaler: StandardScaler instance
        features_list: List of feature column names
        cache_paths: Dict with model cache paths (catboost_path, tabnet_path, etc.)
        market_regime: Current market regime string
        force_retrain: Force retraining even if cache is valid

    Returns:
        dict with trained models and metadata, or None on failure
    """
    import joblib

    if full_df.empty:
        return None

    # Unpack cache paths
    catboost_path = cache_paths['catboost']
    tabnet_path = cache_paths['tabnet']
    tcn_path = cache_paths['tcn']
    elasticnet_path = cache_paths['elasticnet']
    meta_learner_path = cache_paths['meta_learner']
    meta_metadata_path = meta_learner_path.replace('.pkl', '_metadata.json')

    cache_duration_days = PERFORMANCE_CONFIG.get('model_cache_days', 7)

    # --- CACHE CHECK ---
    all_cached = all([
        os.path.exists(catboost_path),
        os.path.exists(tabnet_path),
        os.path.exists(tcn_path),
        os.path.exists(elasticnet_path),
        os.path.exists(meta_learner_path),
        os.path.exists(meta_metadata_path)
    ])

    if not force_retrain and all_cached:
        try:
            model_age_days = (time.time() - os.path.getmtime(meta_learner_path)) / (24 * 3600)
            with open(meta_metadata_path, 'r') as f:
                metadata = json.load(f)

            cached_regime = metadata.get('market_regime', 'Unknown')
            if model_age_days <= cache_duration_days and cached_regime == market_regime:
                print(f"\n[3/4] Loading Cached Diverse Ensemble (Age: {model_age_days:.1f}d, Regime: {market_regime})...")

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Load all models
                catboost_cache = joblib.load(catboost_path)
                result = {
                    'catboost_model': catboost_cache['model'],
                    'imputer': catboost_cache['imputer'],
                    'scaler': catboost_cache['scaler'],
                    'features_list': catboost_cache['features_list'],
                    'tabnet_model': None,
                    'tcn_model': None,
                    'tcn_n_features': 6,
                    'tcn_seq_len': 10,
                    'elasticnet_model': None,
                    'meta_learner': None,
                    'model_trained': True
                }

                if os.path.exists(tabnet_path):
                    tabnet_cache = joblib.load(tabnet_path)
                    result['tabnet_model'] = tabnet_cache['model']

                if os.path.exists(tcn_path):
                    tcn_cache = torch.load(tcn_path, map_location=device, weights_only=False)
                    result['tcn_n_features'] = tcn_cache.get('input_size', 6)
                    result['tcn_seq_len'] = tcn_cache.get('seq_len', 10)
                    from .neural import TCN
                    tcn_model = TCN(input_size=result['tcn_n_features'], num_channels=[32, 32, 16]).to(device)
                    tcn_model.load_state_dict(tcn_cache['model_state'])
                    tcn_model.eval()
                    result['tcn_model'] = tcn_model

                if os.path.exists(elasticnet_path):
                    elasticnet_cache = joblib.load(elasticnet_path)
                    result['elasticnet_model'] = elasticnet_cache['model']

                meta_cache = joblib.load(meta_learner_path)
                result['meta_learner'] = meta_cache['model']

                print(f"  [ENSEMBLE] Loaded cached diverse ensemble (AUC: {metadata.get('ensemble_auc', 'N/A')})")
                print(f"  [ENSEMBLE] Models: {metadata.get('models_available', ['CB'])}")
                return result
            elif cached_regime != market_regime:
                print(f"\n[3/4] Regime Changed ({cached_regime} → {market_regime}), Retraining...")
            else:
                print(f"\n[3/4] Ensemble Expired (Age: {model_age_days:.1f}d > {cache_duration_days}d), Retraining...")
        except Exception as e:
            print(f"\n[3/4] Cache validation failed ({e}), Retraining...")

    # --- TRAINING ---
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    print("\n[3/4] Training Diverse Ensemble (CatBoost + TabNet + TCN + ElasticNet)...")

    # Generate labels
    use_triple_barrier = True
    try:
        price_df = history_mgr.db.get_price_df()
        if not price_df.empty and 'ticker' in full_df.columns:
            tickers = full_df['ticker'].unique().tolist()
            print(f"  [LABELS] Generating Triple-Barrier labels for {len(tickers)} tickers...")

            tb_labels = triple_barrier_labels(
                price_df, tickers,
                holding_period=5,
                pt_multiplier=1.5,
                sl_multiplier=1.0,
                vol_lookback=20
            )

            if not tb_labels.empty:
                latest_labels = tb_labels.sort_values('date').groupby('ticker').last().reset_index()
                latest_labels['target'] = (latest_labels['label'] == 1).astype(int)

                full_df = full_df.merge(
                    latest_labels[['ticker', 'target']],
                    on='ticker', how='left', suffixes=('_old', '')
                )
                full_df['target'] = full_df['target'].fillna(0).astype(int)

                tp_pct = (tb_labels['barrier_hit'] == 'TP').mean() * 100
                sl_pct = (tb_labels['barrier_hit'] == 'SL').mean() * 100
                time_pct = (tb_labels['barrier_hit'] == 'TIME').mean() * 100
                print(f"  [LABELS] Triple-Barrier stats: TP={tp_pct:.1f}% SL={sl_pct:.1f}% TIME={time_pct:.1f}%")
            else:
                use_triple_barrier = False
        else:
            use_triple_barrier = False
    except Exception as e:
        print(f"  [!] Triple-Barrier labeling failed ({e}), using fallback")
        use_triple_barrier = False

    if not use_triple_barrier:
        print(f"  [LABELS] Using simple 5-day return labels (fallback)")
        if 'lagged_return_5d' not in full_df.columns:
            full_df['lagged_return_5d'] = 0.0
        full_df['target'] = (full_df['lagged_return_5d'] > 0.02).astype(int)

    if full_df['target'].nunique() < 2:
        return None

    # Build features_list dynamically if not provided
    if not features_list:
        # Technical features from calculate_technicals()
        tech_feats = ['rsi', 'trend_score', 'volatility', 'sma_alignment', 'divergence_score', 'dist_sma50']
        # Order Flow Imbalance features (v12)
        flow_imbalance_feats = ['clv', 'cmf_20', 'obv_slope', 'vwap_distance']
        # Fractional Differentiation features (v12) - stationary with memory
        frac_diff_feats = ['frac_diff_close']
        # VPIN Flow Toxicity features (v12) - informed trading detection
        vpin_feats = ['vpin', 'vpin_velocity']
        # Options flow features
        flow_feats = ['dp_sentiment', 'net_flow', 'avg_iv', 'net_gamma', 'oi_change', 'dp_total']
        # Temporal momentum features
        temporal_feats = ['gamma_velocity', 'oi_accel']
        # Neural network score
        neural_feats = ['nn_score']

        all_possible = tech_feats + flow_imbalance_feats + frac_diff_feats + vpin_feats + flow_feats + temporal_feats + neural_feats
        features_list = [f for f in all_possible if f in full_df.columns]
        print(f"  [FEATURES] Auto-detected {len(features_list)} features: {features_list}")

    # Prepare features
    X = full_df[features_list]
    y = full_df['target']
    X_clean = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_clean)

    # Configuration
    n_trials = PERFORMANCE_CONFIG.get('catboost_trials', 25)
    max_iterations = PERFORMANCE_CONFIG.get('catboost_max_iterations', 500)
    max_depth = PERFORMANCE_CONFIG.get('catboost_max_depth', 10)
    cv_folds = PERFORMANCE_CONFIG.get('catboost_cv_folds', 5)

    # Detect GPU
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"  [GPU] Detected: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  [GPU] No GPU detected, using CPU")

    print(f"  [LEVEL-1] Training 3 base models ({n_trials} trials each)...")

    # Train base models
    catboost_model, best_cb_params, catboost_auc = train_catboost(
        X_scaled, y, has_gpu, n_trials, max_iterations, max_depth, cv_folds
    )
    if catboost_model is None:
        return None

    tabnet_model, tabnet_auc = train_tabnet(X_scaled, y, has_gpu)

    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    tickers = full_df['ticker'].unique().tolist() if 'ticker' in full_df.columns else []
    tcn_model, tcn_n_features, tcn_seq_len, tcn_auc = train_tcn(
        price_df, full_df, tickers, seq_len=10, has_gpu=has_gpu
    )

    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    elasticnet_model, elasticnet_auc = train_elasticnet(X_scaled, y, cv_folds)

    # Train meta-learner
    meta_learner, model_names, ensemble_auc = train_meta_learner(
        X_scaled, y, catboost_model, tabnet_model, elasticnet_model, cv_folds
    )

    # --- SAVE CACHE ---
    try:
        joblib.dump({
            'model': catboost_model,
            'imputer': imputer,
            'scaler': scaler,
            'features_list': features_list,
            'auc': catboost_auc,
            'params': best_cb_params
        }, catboost_path)

        if tabnet_model is not None:
            joblib.dump({'model': tabnet_model, 'auc': tabnet_auc}, tabnet_path)

        if tcn_model is not None:
            torch.save({
                'model_state': tcn_model.state_dict(),
                'auc': tcn_auc,
                'input_size': tcn_n_features,
                'seq_len': tcn_seq_len
            }, tcn_path)

        if elasticnet_model is not None:
            joblib.dump({'model': elasticnet_model, 'auc': elasticnet_auc}, elasticnet_path)

        joblib.dump({'model': meta_learner, 'model_names': model_names}, meta_learner_path)

        metadata = {
            'timestamp': time.time(),
            'market_regime': market_regime,
            'ensemble_auc': float(ensemble_auc),
            'catboost_auc': float(catboost_auc),
            'tabnet_auc': float(tabnet_auc),
            'tcn_auc': float(tcn_auc),
            'elasticnet_auc': float(elasticnet_auc),
            'gpu_used': has_gpu,
            'models_available': model_names
        }
        with open(meta_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  [ENSEMBLE] All models cached (expires in {cache_duration_days} days or on regime change)")
    except Exception as e:
        print(f"  [!] Ensemble caching failed: {e}")

    return {
        'catboost_model': catboost_model,
        'tabnet_model': tabnet_model,
        'tcn_model': tcn_model,
        'tcn_n_features': tcn_n_features,
        'tcn_seq_len': tcn_seq_len,
        'elasticnet_model': elasticnet_model,
        'meta_learner': meta_learner,
        'imputer': imputer,
        'scaler': scaler,
        'features_list': features_list,
        'model_trained': True
    }
