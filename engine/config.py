# -*- coding: utf-8 -*-
"""
Grandmaster Engine v11.5 - Configuration Constants

All tunable parameters, thresholds, and feature flags for the engine.
Extracted from v11.py for maintainability.
"""

# =============================================================================
# SECTOR MAPPING
# =============================================================================
SECTOR_MAP = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY', 'Industrials': 'XLI', 'Communication Services': 'XLC',
    'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE', 'Utilities': 'XLU'
}

# =============================================================================
# PATTERN DETECTION THRESHOLDS
# =============================================================================

# Bull Flag Configuration (CALIBRATED v9.5+ - production ready)
BULL_FLAG_CONFIG = {
    'pole_min_gain': 0.05,        # 5% minimum pole gain
    'pole_days': 12,              # Days to measure pole
    'flag_max_range': 0.15,       # 15% max consolidation (was 12%, catches BFLY now)
    'flag_days': 10,              # Days of consolidation
    'volume_decline_ratio': 0.95  # Very relaxed - 95% still counts
}

GEX_WALL_CONFIG = {
    'min_support_gamma': 50_000,     # Lowered to 50K for more wall detection
    'min_resist_gamma': -50_000,     # Raised to -50K for more resistance walls
    'proximity_pct': 0.10            # 10% proximity - walls up to 10% away count
}

# Phoenix Reversal Configuration (Extended for Institutional Patterns)
# v10.4: Supports both speculative (2-10 months) and institutional (12-24 months)
# v10.8.1: Fixed consolidation threshold (10% too strict for deep drawdown recoveries)
PHOENIX_CONFIG = {
    'min_base_days': 60,           # Minimum consolidation period (2 months)
    'max_base_days': 730,          # Extended to 24 months for institutional phoenix (was 250)
    'institutional_threshold': 365,  # 12+ months = institutional phoenix (LULU-like patterns)
    'volume_surge_threshold': 1.5, # Volume must exceed 1.5x average
    'rsi_min': 50,                 # RSI must be between 50-70 (not oversold, not overbought)
    'rsi_max': 70,
    'max_drawdown_pct': 0.70,      # Extended to 70% for deep corrections (was 0.35, LULU had 60%)
    'min_consolidation_pct': 0.15, # v10.8.1: Increased from 0.10 to 0.15 (VCP standard)
    'breakout_threshold': 0.03     # Breakout must be at least 3% move
}

REVERSAL_CONFIG = {
    'lookback_days': 45,          # 45 day lookback for downtrend
    'min_days_below_sma': 15,     # Lowered to 15 days - even more lenient
    'dp_proximity_pct': 0.10      # 10% proximity for DP support
}

# =============================================================================
# SMART GATEKEEPER (v11.0)
# =============================================================================
# Dollar-volume liquidity thresholds by market cap tier
# Filters illiquid stocks before expensive pattern analysis
GATEKEEPER_CONFIG = {
    # Dollar-volume thresholds (price × volume)
    'large_cap_threshold': 5_000_000,    # $5M daily dollar-volume for large caps
    'mid_cap_threshold': 2_000_000,      # $2M for mid caps
    'small_cap_threshold': 1_000_000,    # $1M for small caps

    # Market cap classification ($B)
    'large_cap_min': 10_000_000_000,     # $10B+ = large cap
    'mid_cap_min': 2_000_000_000,        # $2B-$10B = mid cap
    # Below $2B = small cap

    # Dark Pool Bypass: High DP inflow bypasses liquidity filter
    'dp_bypass_threshold': 500_000,      # $500K+ DP inflow = allow through

    # Bid-Ask Spread: Max spread for tradeable executions
    'max_spread_pct': 0.005,             # 0.5% max spread

    # Options Liquidity (for options traders)
    'min_options_oi': 500,               # Minimum 500 contracts open interest
}

# =============================================================================
# SOLIDITY SCORE (v11.0)
# =============================================================================
# Detects institutional accumulation during retail exhaustion
# Key insight: High DP activity + narrow price range + declining volume = smart money loading
SOLIDITY_CONFIG = {
    # Consolidation Parameters
    'fib_retracement': 0.382,            # 38.2% Fibonacci level (institutional standard)
    'min_consolidation_days': 20,        # 20+ days of tight range required
    'max_consolidation_range': 0.382,    # Price range within 38.2% of base

    # Volume Decline Detection
    'volume_decline_ratio': 0.70,        # Recent volume < 70% of average = declining
    'volume_lookback_days': 50,          # Compare recent 5d vs 50d average

    # Institutional Flow Thresholds
    'min_dp_total': 10_000_000,          # $10M+ dark pool activity
    'signature_print_bonus': 0.10,       # +10% score for signature prints

    # Scoring Weights
    'weight_in_phoenix': 0.18,           # 18% of final Phoenix score (15-20% range)
    'base_threshold': 0.55,              # v11.2: institutional accumulation must be clear
}

# =============================================================================
# DUAL-RANKING CONFIGURATION (v11.0)
# =============================================================================
# Separate Alpha Momentum and Phoenix Reversal leaderboards
DUAL_RANKING_CONFIG = {
    # Alpha Momentum Leaderboard (continuation plays)
    'alpha_momentum': {
        'min_score': 75,                 # Minimum score to qualify
        'top_n': 25,                     # Top 25 momentum picks
        # Scoring weights for Alpha Momentum
        'weight_trend': 0.30,            # 30% trend/price action
        'weight_ml': 0.25,               # 25% ML ensemble prediction
        'weight_neural': 0.15,           # 15% Hive Mind neural score
        'weight_volume': 0.15,           # 15% volume momentum
        'weight_pattern': 0.15,          # 15% bull flag/continuation patterns
    },

    # Phoenix Reversal Leaderboard (reversal plays)
    'phoenix_reversal': {
        'min_score': 55,                 # v11.2: Lowered from 60 (more lenient)
        'top_n': 25,                     # Top 25 reversal picks
        # Scoring weights for Phoenix (LULU-optimized)
        'weight_solidity': 0.20,         # v11.2: Increased from 0.18
        'weight_duration': 0.20,         # 20% consolidation duration
        'weight_flow': 0.15,             # v11.2: Decreased from 0.20
        'weight_breakout': 0.15,         # 15% breakout confirmation
        'weight_pattern': 0.15,          # 15% reversal patterns (double bottom, etc.)
        'weight_ml': 0.15,               # v11.2: Increased from 0.12
    },
}

# Sector Capping (Risk Management)
MAX_PICKS_PER_SECTOR = 3

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================
PERFORMANCE_CONFIG = {
    'catboost_trials': 25,           # Optuna trials for hyperparameter tuning
    'catboost_max_iterations': 500,  # Max iterations per model
    'catboost_max_depth': 10,        # Max tree depth
    'catboost_cv_folds': 5,          # Cross-validation folds
    'model_cache_days': 7,           # Model cache duration (invalidated on regime change)
    'transformer_epochs': 30,        # Epochs for SwingTransformer
    'max_tickers_to_fetch': 3000,    # Limit ticker downloads for speed
    'price_cache_days': 7,           # Refresh price cache weekly
    'batch_size': 75                 # Batch size for yfinance downloads
}

# =============================================================================
# VALIDATION SUITE (v10.6)
# =============================================================================
# Force-include known institutional phoenix patterns to validate detection
ENABLE_VALIDATION_MODE = True  # Change to False once validated

VALIDATION_SUITE = {
    # Institutional Phoenix (12-24 month bases, activist/turnaround plays)
    'institutional_phoenix': [
        'LULU',  # Elliott $1B stake, 730d base, 60% drawdown, double bottom
        'NVO',   # Novo Nordisk - pharma turnaround, institutional accumulation
    ],
    # Optional: Add other pattern types for comprehensive testing
    'speculative_phoenix': [
        # 'ABC',  # Example: 200-day base, moderate drawdown
    ],
    # Known false positives to test filtering
    'negative_cases': [
        'FCX',   # Freeport-McMoRan - differentiated by recency filter (valid)
        'KGC',   # Kinross Gold - sector beta, filtered by recency
    ]
}

# =============================================================================
# MACRO REGIME WEIGHTS (v10.10)
# =============================================================================
# Dynamic Z-Score Based Adjustments
MACRO_WEIGHTS = {
    # VIX: Penalty if 2.0 std devs above 6-month average
    'vix_z_threshold': 2.0,
    'vix_penalty_per_sigma': 2.0,  # -2 points for every sigma above 2.0

    # TNX: Penalty if 2.0 std devs above 6-month average (Rate Shock)
    'tnx_z_threshold': 2.0,
    'tnx_penalty_per_sigma': 3.0,  # Rates shocks are deadly for swings

    # DXY: Penalty if 2.0 std devs above 6-month average (Dollar Spike)
    'dxy_z_threshold': 2.0,
    'dxy_penalty_per_sigma': 1.5
}

# =============================================================================
# VIX TERM STRUCTURE (PHASE 2)
# =============================================================================
# Advanced volatility analysis for better regime detection
# Reference: @alshfaw analysis - 1.26 is historically bearish for SPY
VIX_TERM_STRUCTURE_CONFIG = {
    # Term Structure Detection (VIX3M / VIX30D ratio)
    'mild_contango_threshold': 1.10,      # VIX3M/VIX 1.10-1.20 = healthy low vol
    'extreme_contango_threshold': 1.20,   # VIX3M/VIX > 1.20 = COMPLACENCY
    'backwardation_threshold': 0.95,      # VIX3M/VIX < 0.95 = fear
    'mild_contango_bonus': 2.0,           # +2 points in healthy contango
    'extreme_contango_penalty': 4.0,      # -4 points when > 1.20 (complacency trap)
    'backwardation_penalty': 5.0,         # -5 points in backwardation (fear)

    # VVIX Divergence Detection
    'vvix_high_threshold': 110,           # VVIX > 110 = elevated vol-of-vol
    'vvix_low_threshold': 85,             # VVIX < 85 = complacency
    'vvix_divergence_lookback': 5,        # Days to check for divergence
    'vvix_divergence_penalty': 4.0,       # Penalty when VVIX diverges bearishly
    'vvix_convergence_bonus': 2.0,        # Bonus when VVIX converges bullishly
}

# =============================================================================
# POSITION SIZING (PHASE 10 - Kelly Criterion)
# =============================================================================
# Kelly Formula: Position = (Edge * P(win) - P(loss)) / Vol_Risk
# Capped at 0.25 * Kelly for safety (quarter-Kelly)
POSITION_SIZING_CONFIG = {
    'kelly_fraction': 0.25,           # Quarter-Kelly for safety
    'max_position_pct': 0.10,         # Max 10% of portfolio per position
    'min_position_pct': 0.01,         # Min 1% of portfolio per position
    'default_win_rate': 0.55,         # Conservative default win rate
    'default_edge': 0.10,             # Conservative default edge (10%)
}

# Risk Tier Matrix: Vol buckets determine position multipliers
RISK_TIER_MATRIX = {
    # nn_score ranges -> vol_bucket -> position multiplier
    'score_tiers': {
        (95, 100): {'<5%': 1.0, '5-6%': 0.75, '>6%': 0.50},   # Highest conviction
        (85, 94):  {'<5%': 0.75, '5-6%': 0.50, '>6%': 0.35},  # High conviction
        (70, 84):  {'<5%': 0.50, '5-6%': 0.35, '>6%': 0.20},  # Medium conviction
    },
    'vol_thresholds': {
        'low': 0.05,    # <5% ATR/Price = low vol
        'medium': 0.06, # 5-6% ATR/Price = medium vol
        # >6% = high vol
    }
}

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    'SECTOR_MAP',
    'BULL_FLAG_CONFIG',
    'GEX_WALL_CONFIG',
    'PHOENIX_CONFIG',
    'REVERSAL_CONFIG',
    'GATEKEEPER_CONFIG',
    'SOLIDITY_CONFIG',
    'DUAL_RANKING_CONFIG',
    'MAX_PICKS_PER_SECTOR',
    'PERFORMANCE_CONFIG',
    'ENABLE_VALIDATION_MODE',
    'VALIDATION_SUITE',
    'MACRO_WEIGHTS',
    'VIX_TERM_STRUCTURE_CONFIG',
    'POSITION_SIZING_CONFIG',
    'RISK_TIER_MATRIX',
]
