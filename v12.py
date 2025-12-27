# -*- coding: utf-8 -*-
"""
SwingEngine v12 - Grandmaster (Modular Architecture)

Modular refactor of v11.5 - core logic in main file, utilities in engine/ package.
Target: ~4500 lines (down from 5600) with modular imports.

Architecture:
1. BACKBONE: SQLite Database (Scalable History).
2. BRAIN: 5x Ensemble Transformer (Hive Mind) + Diverse Ensemble Stack (CatBoost + TabNet + TCN + ElasticNet) + GPU Acceleration.
3. CONTEXT: Enhanced Macro-Regime Awareness (VIX, TNX, DXY) with score weighting.
4. EXECUTION: ATR-based Stop Loss & Take Profit.
5. PATTERNS: 6 Pattern Types - Bull Flag, GEX Wall, Downtrend Reversal, Phoenix, Cup-Handle, Double Bottom.
6. INTERPRETABILITY: Human-readable explanation generator for each signal.
7. RISK: Sector capping to prevent over-concentration.
8. PERFORMANCE: Ensemble caching, GPU acceleration, early stopping, configurable parameters.
9. SIZING: Kelly Criterion position sizing with volatility-based risk tiers.
10. v11.0 DUAL-RANKING: Separate Alpha Momentum and Phoenix Reversal leaderboards.
11. v11.0 SOLIDITY: Institutional accumulation detection (38.2% Fib consolidation + declining volume).
12. v11.0 GATEKEEPER: Dollar-volume liquidity pre-filtering with DP bypass.

Changelog from v8.4:
- Added bull flag pattern detection (consolidation after strong moves)
- Added GEX wall scanner (gamma exposure support/resistance levels)
- Added downtrend reversal detection (months-long downtrend + DP support)
- Added explanation generator (human-readable reasoning for each pick)
- Added sector capping (max 3 picks per sector per strategy)
- Enhanced macro weighting (DXY/TNX/VIX influence individual scores)
- Removed hard Colab dependency (works locally or in Colab)

Changelog v9.5 (Calibration):
- Relaxed bull flag thresholds (5% pole min, 15% max range) for more detections
- Fixed GEX wall protection score (250K divisor for proper 75K+ = protected)
- Added GEX debug output showing candidate/strike data overlap
- Performance config: reduced transformer epochs (50->30), CatBoost trials (30->15)
- Configurable batch sizes and ticker download limits
- TPU/GPU/CPU auto-detection with torch_xla support

Changelog v10.0 (Position Sizing):
- Added Kelly Criterion position sizing (quarter-Kelly for safety)
- Added Risk Tier Matrix (HIGH/MEDIUM/LOW conviction based on nn_score + vol)
- Added volatility buckets (<5%, 5-6%, >6% ATR/Price)
- Position sizing integrated into all strategy outputs
- Displays position_pct, shares, and risk_tier for each pick

Changelog v10.1 (Phoenix + Reliability):
- ADDED: Phoenix Reversal Strategy (6-12 month base breakouts with volume surge)
- ENHANCED: CatBoost AUC improvement (25 trials, max_iter=500, depth=10, CV=5)
- FIXED: VIX/TNX/DXY reliability with NaN checking and retry logic (3 retries)
- IMPROVED: Fetch success tracking with detailed error handling and logging
- OPTIMIZED: Border count tuning in CatBoost for better model performance
- Updated output filename to v10_grandmaster.csv

Changelog v10.2 (Production Optimization - Quick Wins):
- PERFORMANCE: Model caching with regime-aware invalidation (7-day default, 50-70min savings)
- PERFORMANCE: Early stopping in CatBoost (early_stopping_rounds=50)
- PATTERNS: Multi-layer phoenix scoring system (6 layers: duration, volume, RSI, breakout, drawdown, DP)
- PATTERNS: Added Cup-and-Handle detection (U-shaped recovery + handle consolidation)
- PATTERNS: Added Double Bottom detection (dual support test + resistance breakout)
- PATTERNS: Enhanced phoenix threshold from boolean AND to weighted scoring (0.60 threshold)
- PATTERNS: Detailed point-based explanations showing strength/weakness breakdown
- Expected runtime: 83min → 20-30min (first run trains, subsequent runs load cache)
- Expected phoenix detections: 0 → 3-8 per run (with relaxed scoring)

Changelog v10.3 (Ensemble Stacking + GPU Acceleration):
- ML ARCHITECTURE: Replaced single CatBoost with ensemble stacking (3 models + meta-learner)
- ML MODELS: CatBoost + LightGBM + XGBoost with LogisticRegression meta-learner (v10.3)
- GPU ACCELERATION: All 3 base models use GPU when available (task_type='GPU', device='gpu', device='cuda')

Changelog v11.3 (Phase 3: ML Ensemble Overhaul - Architectural Diversity):
- ML MODELS: Replaced redundant boosting (LightGBM/XGBoost) with diverse architectures
- ENSEMBLE: CatBoost (tree splits) + TabNet (attention) + TCN (temporal) + ElasticNet (linear)
- TCN: Temporal Convolutional Network for sequential pattern detection (dilated causal convolutions)
- TABNET: Attention-based feature interaction learning (sparsemax/entmax masks)
- ELASTICNET: L1+L2 regularized linear baseline (overfitting sanity check)
- TARGET AUC: 0.945-0.955 through cognitive diversity vs 0.93-0.94 redundant boosting
- PERFORMANCE: Ensemble caching system (4 model caches with regime-aware invalidation)
- PERFORMANCE: Cross-validated stacking predictions for meta-learner training
- QUALITY: Expected AUC improvement from 0.92-0.93 → 0.95+ with ensemble diversity
- GPU DETECTION: Automatic CUDA detection with fallback to CPU
- LOGGING: Detailed model weights and individual AUC scores for transparency
- Expected runtime: 34min → 18-22min with GPU (first run trains all 3, subsequent runs load cache)
- Expected AUC: 0.92 → 0.95+ with ensemble stacking

Changelog v11.4 (Phase 4: Feature Engineering - Triple-Barrier + TCN Fix):
- LABELS: Triple-Barrier Labeling replaces simple binary labels for realistic trade outcomes
  - TP barrier: 1.5x ATR take-profit target
  - SL barrier: 1.0x ATR stop-loss level
  - TIME barrier: 5-day holding period max
  - Labels reflect which barrier was hit first (TP=win, SL=loss)
- TCN FIX: Real temporal sequences replace fake repeated snapshots
  - Previous bug: Same feature vector repeated N times (no temporal info)
  - Now: Actual day-by-day feature evolution (returns, volatility, RSI, momentum)
  - Features: returns, volatility, volume_ratio, rsi, momentum_5d, momentum_10d
- INFERENCE: TCN generates per-ticker sequences from price history
- TARGET AUC: 0.92 → 0.94+ with proper labeling and real sequences

Changelog v10.4 (Institutional Phoenix + Pattern Synergy - LULU-Inspired):
- PHOENIX EXTENDED: Base duration extended from 250 days → 730 days (24 months)
- PHOENIX TIERS: Speculative (2-12 months) vs Institutional (12-24 months) phoenix patterns
- PHOENIX SCORING: Institutional phoenix (365-730 days) gets FULL base duration score (0.25)
- DRAWDOWN EXTENDED: Acceptable drawdown 35% → 70% for deep institutional corrections (LULU: 60%)
- DRAWDOWN SCORING: 50-70% drawdowns score HIGHER for institutional patterns (deep value opportunity)
- DARK POOL MAGNITUDE: Logarithmic scaling for mega-prints ($50M+, $500M+, $1B+ institutional activism)
- DARK POOL BONUS: $500M+ prints (Elliott/LULU-level) get +0.25 total DP score (vs 0.15 baseline)
- PATTERN SYNERGY: Phoenix + Double Bottom = +8 point bonus (LULU pattern recognition)
- PATTERN SYNERGY: Phoenix + Cup-Handle = +6 point bonus (institutional continuation)
- PATTERN SYNERGY: Bull Flag + GEX Wall = +5 point bonus (momentum + support)
- INSTITUTIONAL FOCUS: Catches large-cap ($5B+) activist plays with extended accumulation periods
- Expected improvement: Catch LULU-like patterns that were previously missed
- Rationale: Real-world validation showed BHR flagged but LULU (Elliott $1B stake) missed

Changelog v10.5 (Oracle Reliability Fix - Data Integrity):
- ROOT CAUSE: yfinance bulk downloads failing silently (63.5% success rate → 36.5% data loss)
- ORACLE CHUNKING: Reduced batch size 75 → 50 tickers per chunk (API limit compliance)
- RATE LIMITING: 1.5s delay between chunks with exponential backoff on errors (prevents bans)
- INDIVIDUAL RETRY: Failed tickers get individual fetch retry (yfinance quirk workaround)
- DATA INTEGRITY GATE: <90% fetch success → WARNING + user notification (no silent failures)
- PROGRESS FEEDBACK: Chunk-by-chunk progress reporting for transparency
- EXPECTED IMPROVEMENT: 63.5% → 95%+ Oracle fetch success rate
- PHOENIX IMPACT: LULU-like patterns now detectable (data present → analysis works)
- Rationale: Previous sessions found phoenix logic correct but no data to analyze

Changelog v10.6 (Validation Mode Fix - CRITICAL):
- ROOT CAUSE CORRECTED: Previous diagnosis missed the REAL blocker
- THREE DATA STAGES: Oracle (781) → Fetch (3000) → Patterns (top 75 only)
- FATAL FLAW: Validation mode added LULU with zeros → Low ML score → Never in top 75
- FIX: Force validation tickers into top_candidates (bypass ML ranking filter)
- VALIDATION_SUITE: Moved to module-level constant for cross-function access
- DEBUG OUTPUT: Shows LULU's actual ML score, rank, data status, and phoenix result
- PATTERN CHUNKING: Applied 50-ticker chunking to pattern downloads too (safety)
- EXPECTED: LULU now reaches pattern analysis → Phoenix detection can run
- Rationale: Validation mode adds ticker but NOT data → Pattern analysis never reached

Changelog v10.6.1 (Data Sanitization - yfinance Header Leak Fix):
- BUG: yfinance batch downloads leak header "Ticker" into data rows
- CRASH: pd.to_datetime("Ticker") fails → ValueError stops engine
- FIX 1: prepare_supervised_data uses errors='coerce' + dropna for robust parsing
- FIX 2: sync_prices sanitizes data BEFORE DB insert (defense-in-depth)
- REMOVES: Corrupt rows with "Ticker" in date column automatically cleaned
- Rationale: yfinance quirk when downloading in chunks/groups

Changelog v10.7 (Flow-Adjusted Dynamic Scoring Model):
- ARCHITECTURE: Flow-Adjusted Dynamic Scoring replaces hard thresholds
- CONCEPT: High institutional flow earns "forgiveness points" that expand limits
- EXAMPLE: LULU with $1B Elliott stake can have RSI 80 and still pass
- NEW METHOD: calculate_flow_factor() computes institutional conviction (0.0-1.0)
- FLOW COMPONENTS: Volume intensity + Dark Pool activity + Signature prints + Options flow
- DYNAMIC RSI: Base 50-70 expands to 40-85 with max flow factor
- DYNAMIC THRESHOLD: Phoenix threshold 0.60 → 0.45 with max flow factor
- FLOW BONUS: High flow adds 5-15 points directly to composite score
- DATA FIX: Extended all downloads from 3mo to 2y (504 trading days)
- MACRO FIX: get_market_regime with cooldown, individual fallback, and sanitization
- MACRO COOLDOWN: 2s initial delay + exponential backoff between retries
- MACRO FALLBACK: If batch fails, tries individual ticker fetches with 1s cooldown
- Rationale: Binary thresholds rejected LULU (RSI 78) despite $1B institutional stake

Changelog v10.8 (Anti-Throttling Oracle - Yahoo Rate Limit Defense):
- PROBLEM: Yahoo Finance throttling causes 40% fetch success rate (too many requests)
- LEVEL 1 FIX: User-Agent spoofing (Chrome browser headers) to appear as normal browser
- LEVEL 2 FIX: Random delays 3-6s between chunks + smaller chunk size (50 → 20)
- LEVEL 3 FIX: Incremental updates - only fetch data newer than last DB entry
- WEEKEND AWARENESS: Skips sync on weekends when markets are closed (no new data)
- SMART CACHE CHECK: Validates existing data freshness before full refresh
- SESSION POOLING: Persistent requests session with connection pooling
- EXPECTED: 40% → 90%+ Oracle success rate via stealth + efficiency
- Rationale: Repeated large bulk requests triggered Yahoo's rate limiter

Changelog v10.8.1 (Phoenix Detection Fix - LULU-Critical):
- BUG 1 FIX: Phoenix lookback was truncated to 60 days (min_days)
  - Previous: days_in_base capped at 60, institutional phoenix (365+) IMPOSSIBLE
  - Now: Looks at FULL lookback period (up to 730 days)
- BUG 2 FIX: Consolidation threshold was too strict (10%)
  - Industry standard: 15-18% for VCP/flat base patterns (TraderLion/Minervini)
  - For deep drawdown recoveries (60%+ drop like LULU): 20-25% is acceptable
  - New: Dynamic threshold 15% base, +10% for deep drawdowns, +5% for high flow
- DYNAMIC CONSOLIDATION: consolidation_threshold = 0.15 + drawdown_bonus + flow_bonus
  - drawdown_bonus: Up to 10% extra for deep drawdowns (base_range * 0.15)
  - flow_bonus: Up to 5% extra for institutional flow (flow_factor * 0.05)
  - Capped at 30% maximum (prevents false positives)
- DEBUG OUTPUT: Phoenix analysis now prints detailed debug for validation tickers
- EXPECTED: LULU should now qualify for Phoenix detection with institutional scoring
- Rationale: LULU had 60% drawdown + ~400 day base but failed 10% consolidation test

Changelog v10.9 (Alpaca Data Layer - Production-Grade Pipeline):
- ARCHITECTURE: Migrated from unstable yfinance to Alpaca Market Data API
- RELIABILITY: Alpaca provides 99.9%+ uptime vs Yahoo's ~75% success rate
- SPEED: Alpaca batch API is 3-5x faster than yfinance for bulk downloads
- COVERAGE: Alpaca handles all stocks/ETFs; yfinance kept only for indices (^VIX, ^TNX)
- NEW FUNCTION: fetch_alpaca_batch() for efficient batched price history
- REFACTORED: sync_prices() now uses Alpaca for Oracle data
- REFACTORED: enrich_market_data() now uses Alpaca for technical fetches
- REFACTORED: Pattern downloads now use Alpaca for 2y history
- PRESERVED: get_market_regime() still uses yfinance for macro indices
- PREREQUISITE: Requires ALPACA_API_KEY and ALPACA_SECRET_KEY
- EXPECTED: 75% → 99%+ data fetch success rate
- Rationale: Yahoo Finance throttling made production use unreliable

Changelog v10.10 (Dynamic Macro Intelligence - Z-Score Adaptation):
- MACRO INTELLIGENCE: Replaced static thresholds with Dynamic Rolling Z-Scores
- ADAPTIVE REGIME: Engine self-calibrates to "New Normal" market conditions (e.g., high-rate eras)
- STATISTICAL BASELINE: Fetches 1-year macro history to calculate 6-month rolling averages/std-devs
- VISIBILITY: Macro status now reports Z-Scores (e.g., "VIX -1.0σ") for context
- DATA HYGIENE: Improved ticker sanitization to exclude warrants ("+") and zombies
- ORACLE EFFICIENCY: Smart weekend-skipping logic (prevents redundant fetches if DB exists)
- EXPECTED: Fewer false positives during high-volatility regimes; robust handling of structural market shifts

Changelog v11.0 (Dual-Ranking Architecture - Phase 1 - LULU Fix):
- CRITICAL FIX: LULU ranked 118th despite 0.96 phoenix score - momentum bias exposed
- DUAL-RANKING: Separate Alpha Momentum vs Phoenix Reversals leaderboards (not competing)
- ALPHA MOMENTUM: Trend score + RSI + price action + volume momentum (continuation plays)
- PHOENIX REVERSAL: Consolidation duration + solidity + breakout + institutional flow (reversal plays)
- SOLIDITY SCORE: Institutional accumulation detection (38.2% Fib consolidation + declining volume)
  - Detects "smart money loading" while retail loses interest
  - Price range within 38.2% of base (institutional Fibonacci level)
  - 20+ day consolidation period required
  - Dark pool activity > $10M threshold
  - Declining retail volume confirmation
  - Weight: 15-20% of final Phoenix score
- SMART GATEKEEPER: Dollar-volume liquidity pre-filtering
  - Large-cap: $5M daily dollar-volume required
  - Mid-cap: $2M daily dollar-volume required
  - Small-cap: $1M daily dollar-volume required
  - DP BYPASS: $500K+ dark pool inflow bypasses liquidity filter (early accumulation)
  - Bid-ask spread check: <= 0.5% for tradeable executions
- EXPECTED: LULU moves from #118 to top 25 Phoenix Reversals where it belongs
- BUG FIX: Leaderboard display regenerated AFTER position sizing (position_pct/risk_tier available)

Changelog v11.2 (Phoenix Calibration + Accumulation Fix):
- CRITICAL BUG FIX: Pattern bonuses were OVERWRITING each other instead of accumulating!
  - Root cause: Each bonus used row.get() which reads ORIGINAL value, not accumulated
  - Example: phoenix_boost(42) + solidity_bonus(9) + duration_bonus(12) → only 12 survived
  - FIX: Use local accumulators (phoenix_accum, alpha_accum, etc.) with single write at end
- SCORING REBALANCE: Increased pattern multipliers for achievable thresholds
  - phoenix_boost: 25 → 40 (60% increase) for detected phoenix patterns
  - solidity_boost: 15 → 20 (33% increase) for institutional bases
  - legacy phoenix bonus: 15 → 25 for trend/ambush scores
- THRESHOLD LOWERED: Phoenix min_score 60 → 55 (more lenient qualification)
- WEIGHT ADJUSTMENTS: Rebalanced Phoenix Reversal scoring formula
  - weight_ml: 0.12 → 0.15 (slight ML increase for base points)
  - weight_solidity: 0.18 → 0.20 (increased institutional detection weight)
  - weight_flow: 0.20 → 0.15 (early accumulation is quiet by design)
- NEW FEATURE: Institutional Duration Bonus (12+ month bases)
  - Adds up to 20 points for bases ≥365 days (LULU: 445d → +12.2 pts)
  - Rewards patience in extended consolidation periods
- MATH VALIDATION: LULU now properly accumulates ALL bonuses (~70+ pts)
- EXPECTED RESULTS: 6-10 phoenix qualifications per run (vs 0 in v11.0)
- RATIONALE: v11.0 detected patterns correctly but scoring was miscalibrated
  - Perfect phoenix (1.0 score) + average ML (50) = only 31 pts (missed 60 threshold)
  - v11.2 fixes: same pattern now scores 65+ pts with proper multipliers

Changelog v11.2 (False Positive Prevention - MU Fix):
- PROBLEM: MU ranked #1 on Phoenix Leaderboard with 99.9 score but only 0.40 solidity
  - MU is a momentum play (near 52-week high), not a reversal from accumulation
  - Heavy dark pool ($1.1B) + bullish gamma inflated score despite weak accumulation pattern
- FIX 1: SOLIDITY GATE - Hard penalty for low solidity scores
  - Raised base_threshold: 0.50 → 0.55 (clearer accumulation required)
  - If solidity < 0.55 AND phoenix_score > 0: Apply 70% penalty to phoenix_accum
  - Effect: MU (0.40 solidity) score drops from 99.9 to ~30 (below 55 threshold)
  - Effect: LULU (0.70 solidity) remains at 99.9 (passes threshold)
- FIX 2: MOMENTUM FILTER - 52-week high proximity check
  - Calculate pct_from_52w_high for each phoenix candidate
  - If stock is within 10% of 52-week high: Apply 50% phoenix penalty
  - Boost alpha_momentum_score instead (correct leaderboard placement)
  - Rationale: True phoenix reversals emerge from BASES, not from stocks at highs
- EXPECTED RESULTS:
  - MU: Removed from Phoenix Leaderboard (low solidity + near 52w high)
  - LULU: Remains top 3 on Phoenix Leaderboard (high solidity + recovering from base)
  - False positive rate: Expected reduction of 60-80%

Changelog v11.2 Phase 2 (VIX Term Structure + VVIX Divergence):
- NEW FEATURE: VIX Term Structure Analysis for regime detection
  - Fetches ^VIX3M (3-month VIX futures) and ^VVIX (volatility of VIX)
  - Calculates term structure ratio: VIX3M / VIX
  - CONTANGO (ratio > 1.05): Bullish signal, +3 points macro adjustment
  - BACKWARDATION (ratio < 0.95): Fear signal, -5 points macro adjustment
  - New regime: "Fear (Backwardation)" when VIX in backwardation + VIX_z > 1.0
- NEW FEATURE: VVIX Divergence Detection for vol spike prediction
  - Tracks 5-day VVIX and VIX changes for divergence patterns
  - BEARISH DIVERGENCE: VVIX rising (>5%) + VIX falling (<-2%) = vol spike incoming
    - Applies -4 points penalty (warns of imminent volatility)
  - BULLISH CONVERGENCE: VVIX falling (<-5%) + VIX rising (>2%) = vol spike fading
    - Applies +2 points bonus (vol normalizing)
  - VVIX Elevated warning when VVIX > 110 (high vol-of-vol)
- ENHANCED OUTPUT: Macro regime now displays term structure and VVIX status
  - Example: "VIX3M: 22.50 | Term Structure: CONTANGO (1.08) | VVIX: 95"
- CONFIG: VIX_TERM_STRUCTURE_CONFIG with tunable thresholds
- EXPECTED RESULTS: Better regime-aware position sizing during vol regime shifts

Changelog v11.2 Phase 2.1 (Extreme Contango Fix - @alshfaw Analysis):
- CRITICAL FIX: Extreme contango (>1.20) is now BEARISH, not bullish
  - Reference: @alshfaw analysis - 1.26 historically precedes SPY corrections
  - 1.26 brought markets down in Aug, Sep, Oct, Nov, Dec 2024
- NEW INTERPRETATION:
  - Normal (0.95-1.10): Neutral, no adjustment
  - Mild contango (1.10-1.20): Healthy low vol, +2 points
  - EXTREME contango (>1.20): COMPLACENCY WARNING, -4 points
  - Backwardation (<0.95): Fear/panic, -5 points
- NEW REGIME: "Complacency (Extreme Contango)" when ratio > 1.20
- CURRENT STATUS (Dec 2025): VIX3M/VIX at 1.26 = EXTREME CONTANGO (bearish)
- EXPECTED: Score adjustment now -4.0 instead of +3.0 (7 point swing)

Changelog v11.5.2 (Momentum Filter Fix - NVO Restoration):
- CRITICAL FIX: High solidity now BYPASSES momentum penalty
  - NVO (solidity 0.70) was incorrectly filtered despite valid institutional accumulation
  - Root cause: Momentum filter penalized ALL stocks near 52w high, including valid breakouts
- NEW LOGIC: Solidity gate on momentum filter
  - Only penalize if solidity < 0.55 (weak accumulation signal)
  - High solidity (>= 0.55) = institutional conviction validates breakout thesis
  - Example: NVO with 0.70 solidity now PASSES, MU with 0.40 solidity still FILTERED
- AFFECTED CHECKS:
  - 52w high proximity: Only penalize LOW solidity candidates near highs
  - 52w low distance: Only penalize LOW solidity candidates far from lows
- EXPECTED: NVO restored to Phoenix Leaderboard alongside LULU

Changelog v11.5.3 (Sector Momentum Filter - FCX/KGC Fix):
- CRITICAL FIX: Sector momentum now penalizes phoenix regardless of solidity
  - FCX/KGC (precious metals) have high solidity but XLB sector is at ATH
  - This is sector BETA, not individual stock ALPHA
  - High solidity in a hot sector = institutions chasing momentum, not finding value
- NEW FILTER: Sector 52-week high proximity check
  - Fetches 1-year data for all sector ETFs (XLK, XLF, XLV, XLB, etc.)
  - If sector within 10% of ATH: Apply 60% phoenix penalty (sector beta)
  - If sector within 15% of ATH: Apply 40% phoenix penalty (moderate)
  - Boosts alpha_score instead (correct leaderboard placement)
- EXAMPLES:
  - FCX (Basic Materials/XLB): XLB at ATH → heavy penalty → moved to Alpha
  - KGC (Basic Materials/XLB): XLB at ATH → heavy penalty → moved to Alpha
  - NVO (Healthcare/XLV): XLV not at ATH → NO penalty → stays on Phoenix
  - LULU (Consumer Cyclical/XLY): Sector-relative check passes → stays on Phoenix
- EXPECTED: FCX/KGC filtered from Phoenix, NVO/LULU remain
- NOTE: v11.5.3 REPLACED by v11.5.4 - absolute ATH approach caused collateral damage

Changelog v11.5.4 (Sector Beta Filter - Relative Performance):
- REPLACED v11.5.3: Absolute sector ATH check was too broad
  - Problem: In bullish market, multiple sectors near ATH → penalized valid phoenix too
  - LULU dropped from 99.9 to 58.4 despite being true phoenix (XLY also near ATH)
- NEW APPROACH: Stock vs Sector RELATIVE performance
  - Compares stock's 6-month return to sector ETF's 6-month return
  - True phoenix = stock OUTPERFORMING sector (individual alpha)
  - Sector beta = stock just MATCHING sector (riding the wave)
- PENALTY TIERS (based on relative performance):
  - Underperforming sector (< -5%): 60% penalty - definitely not a leader
  - Matching sector (-5% to 0%): 50% penalty - likely beta play
  - Barely outperforming (0% to 5%): 30% penalty - mild concern
  - Outperforming (> 5%): NO penalty - true individual alpha
- EXAMPLES:
  - LULU: If up 30% while XLY up 10% → 20% alpha → NO PENALTY → stays on Phoenix
  - FCX: If up 25% while XLB up 25% → 0% alpha → PENALTY → moved to Alpha Momentum
- DATA: Fetches 6-month returns for all sector ETFs (XLK, XLF, XLV, XLB, etc.)
- EXPECTED: True individual reversals (LULU, NVO) pass; sector beta (FCX, KGC) filtered
"""

!pip uninstall -y alpaca-trade-api torch_xla 2>/dev/null
!pip install alpaca-py yfinance "websockets>=13.0" --upgrade --quiet
!pip install pytorch-tabnet optuna catboost --quiet
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

# =============================================================================
# COLAB BOOTSTRAP - Auto-creates engine/ module files if missing/corrupted
# =============================================================================
COLAB_ENV = False
COLAB_BASE = None
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    COLAB_ENV = True
    COLAB_BASE = '/content/drive/My Drive/colab'

    # Ensure engine/ folder exists (must be uploaded or git cloned)
    engine_dir = os.path.join(COLAB_BASE, 'engine')
    if not os.path.exists(engine_dir):
        raise FileNotFoundError(
            f"Engine modules not found at {engine_dir}. "
            "Please upload the 'engine/' folder to your Google Drive colab folder, "
            "or git clone the repo."
        )

    # Add to path for imports
    if COLAB_BASE not in sys.path:
        sys.path.insert(0, COLAB_BASE)
    print(f"  [COLAB] Engine modules loaded from {engine_dir}")
except ImportError:
    pass  # Running locally

# =============================================================================
# ENGINE MODULES (v12 Modular Architecture)
# =============================================================================
from engine import (
    SECTOR_MAP, BULL_FLAG_CONFIG, GEX_WALL_CONFIG, PHOENIX_CONFIG,
    REVERSAL_CONFIG, GATEKEEPER_CONFIG, SOLIDITY_CONFIG, DUAL_RANKING_CONFIG,
    REGIME_WEIGHT_ADJUSTMENTS,  # v12: Regime-adaptive scoring
    MAX_PICKS_PER_SECTOR, PERFORMANCE_CONFIG, ENABLE_VALIDATION_MODE,
    VALIDATION_SUITE, MACRO_WEIGHTS, VIX_TERM_STRUCTURE_CONFIG,
    POSITION_SIZING_CONFIG, RISK_TIER_MATRIX,
    SwingTransformer, TemporalBlock, TCN,
    get_device, configure_yfinance_session, YF_SESSION,
    is_weekend, get_market_last_close_date, Logger,
    sanitize_ticker_for_alpaca, fetch_alpaca_batch,
    TitanDB, HistoryManager,
    triple_barrier_labels, prepare_tcn_sequences,
    # Pattern detection functions (v12 modular)
    detect_bull_flag as _detect_bull_flag,
    find_gex_walls as _find_gex_walls,
    detect_downtrend_reversal as _detect_downtrend_reversal,
    calculate_flow_factor as _calculate_flow_factor,
    calculate_solidity_score as _calculate_solidity_score,
    detect_phoenix_reversal as _detect_phoenix_reversal,
    detect_cup_and_handle as _detect_cup_and_handle,
    detect_double_bottom as _detect_double_bottom,
    apply_smart_gatekeeper as _apply_smart_gatekeeper,
    # ML training functions (v12 modular)
    train_ensemble as _train_ensemble,
    # Report generation (v12 modular)
    generate_full_report as _generate_full_report,
    # Transformer/Hive Mind (v12 modular)
    train_hive_mind as _train_hive_mind,
)

ALPACA_API_KEY = 'PK3D25CFOYT2Z5F6DW54XKQXOO'
ALPACA_SECRET_KEY = 'DczbobRsFCUPinP9QsByBzLf6sGLHdcf1T7P3SGfo7uK'

# Initialize Client (New SDK)
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# =============================================================================
# ML IMPORTS
# =============================================================================
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, ElasticNet
import optuna
from catboost import CatBoostClassifier, Pool
# Phase 3: Removed LightGBM/XGBoost for architectural diversity
# import lightgbm as lgb  # Replaced by TabNet (attention-based)
# import xgboost as xgb   # Replaced by TCN (temporal CNN)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Phase 3: TabNet for attention-based feature interactions
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("  [WARN] pytorch-tabnet not installed. Run: pip install pytorch-tabnet")

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Device detection (imported from engine.utils)
device, device_name = get_device()

# =============================================================================
# CONFIGURATION - Now imported from engine/config.py
# =============================================================================
# All config constants (SECTOR_MAP, PHOENIX_CONFIG, DUAL_RANKING_CONFIG, etc.)
# are now imported from engine.config at the top of this file.
# To modify configuration, edit: engine/config.py (~250 lines)
# This enables faster iteration - only copy config.py to Colab for config changes.

# =============================================================================
# DATA PREPARATION - Imported from engine/data_prep.py
# =============================================================================
# triple_barrier_labels and prepare_tcn_sequences are now imported from engine.data_prep

# =============================================================================
# DATA LOADING - Imported from engine/data_loader.py
# =============================================================================
# TitanDB and HistoryManager are now imported from engine.data_loader

# =============================================================================
# NEURAL NETWORKS - Imported from engine/neural.py
# =============================================================================
# SwingTransformer, TemporalBlock, TCN are now imported from engine.neural

# =============================================================================
# SWING TRADING ENGINE
# =============================================================================
class SwingTradingEngine:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.model_path = os.path.join(self.base_dir, "grandmaster_cat_v8.pkl")
        self.transformer_path = os.path.join(self.base_dir, "grandmaster_transformer_v8.pth")

        # Phase 3: Diverse ensemble model paths (CatBoost + TabNet + TCN + ElasticNet)
        self.catboost_path = os.path.join(self.base_dir, "ensemble_catboost_v11.pkl")
        self.tabnet_path = os.path.join(self.base_dir, "ensemble_tabnet_v11.pkl")
        self.tcn_path = os.path.join(self.base_dir, "ensemble_tcn_v11.pth")
        self.elasticnet_path = os.path.join(self.base_dir, "ensemble_elasticnet_v11.pkl")
        self.meta_learner_path = os.path.join(self.base_dir, "ensemble_meta_v11.pkl")

        self.scaler = StandardScaler()
        self.nn_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.features_list = []
        self.full_df = pd.DataFrame()
        self.price_cache_file = os.path.join(self.base_dir, "price_cache_v79.csv")

        # Phase 3: Diverse ensemble models (architectural diversity > quantity)
        self.catboost_model = None   # Tree-based (splits)
        self.tabnet_model = None     # Attention-based (feature interactions)
        self.tcn_model = None        # Temporal CNN (sequential patterns)
        self.elasticnet_model = None # Linear regularized (sanity check)
        self.meta_learner = None
        self.model = None  # Keep for backward compatibility
        self.nn_model = None
        self.model_trained = False

        self.spy_metrics = {'return': 0.0, 'rsi': 50.0, 'trend': 0.0}
        self.sector_data = {}
        self.sector_6m_returns = {}  # v11.5.4: Sector ETF 6-month returns for relative performance filter
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
        """
        v10.10: Dynamic Regime Adaptation using Rolling Z-Scores.
        Self-calibrates to market conditions (High VIX is relative, not absolute).
        """
        print("\n[TITAN] Assessing Macro Regime (Dynamic Z-Score Mode)...")

        default_macro = {
            'vix': 20.0, 'tnx': 4.0, 'dxy': 100.0,
            'spy_trend': True, 'spy_return': 0.0,
            'adjustment': 0, 'regime_details': ['Data unavailable'],
            # v11.2 Phase 2: VIX Term Structure defaults
            'vix3m': 22.0, 'term_structure_ratio': 1.0, 'term_structure': 'neutral',
            'vvix': 90.0, 'vvix_divergence': 'none'
        }

        try:
            # 1. Fetch 1 Year of history to establish statistical baseline
            # v11.2 Phase 2: Added VIX3M and VVIX for term structure analysis
            tickers = ['^VIX', '^VIX3M', '^VVIX', '^TNX', 'DX-Y.NYB', 'SPY']
            data = yf.download(tickers, period="1y", progress=False, threads=False)

            # Handle MultiIndex columns if necessary
            if isinstance(data.columns, pd.MultiIndex):
                close_df = data.xs('Close', axis=1, level=0)
            else:
                close_df = data['Close'] if 'Close' in data.columns else data

            # Extract Series
            vix_series = close_df['^VIX'].dropna()
            tnx_series = close_df['^TNX'].dropna()
            dxy_series = close_df['DX-Y.NYB'].dropna()
            spy_series = close_df['SPY'].dropna()

            # v11.2 Phase 2: Extract VIX3M and VVIX series
            vix3m_series = close_df['^VIX3M'].dropna() if '^VIX3M' in close_df.columns else None
            vvix_series = close_df['^VVIX'].dropna() if '^VVIX' in close_df.columns else None

            # Get Current Values (Last available)
            vix_curr = float(vix_series.iloc[-1])
            tnx_curr = float(tnx_series.iloc[-1])
            dxy_curr = float(dxy_series.iloc[-1])
            spy_curr = float(spy_series.iloc[-1])

            # v11.2 Phase 2: Get VIX3M and VVIX current values
            vix3m_curr = float(vix3m_series.iloc[-1]) if vix3m_series is not None and len(vix3m_series) > 0 else vix_curr * 1.1
            vvix_curr = float(vvix_series.iloc[-1]) if vvix_series is not None and len(vvix_series) > 0 else 90.0

            # 2. Calculate Z-Scores (6-month rolling window = 126 trading days)
            window = 126

            def calculate_z_score(series, current_val):
                if len(series) < window: return 0.0
                rolling_mean = series.rolling(window=window).mean().iloc[-1]
                rolling_std = series.rolling(window=window).std().iloc[-1]
                if rolling_std == 0: return 0.0
                return (current_val - rolling_mean) / rolling_std

            vix_z = calculate_z_score(vix_series, vix_curr)
            tnx_z = calculate_z_score(tnx_series, tnx_curr)
            dxy_z = calculate_z_score(dxy_series, dxy_curr)

            # 3. Calculate Dynamic Penalties
            macro_adjustment = 0
            regime_details = []

            # VIX Penalty
            if vix_z > MACRO_WEIGHTS['vix_z_threshold']:
                penalty = (vix_z - MACRO_WEIGHTS['vix_z_threshold']) * MACRO_WEIGHTS['vix_penalty_per_sigma']
                macro_adjustment -= penalty
                regime_details.append(f"VIX Stress (+{vix_z:.1f}σ)")

            # TNX Penalty
            if tnx_z > MACRO_WEIGHTS['tnx_z_threshold']:
                penalty = (tnx_z - MACRO_WEIGHTS['tnx_z_threshold']) * MACRO_WEIGHTS['tnx_penalty_per_sigma']
                macro_adjustment -= penalty
                regime_details.append(f"Rate Shock (+{tnx_z:.1f}σ)")

            # DXY Penalty
            if dxy_z > MACRO_WEIGHTS['dxy_z_threshold']:
                penalty = (dxy_z - MACRO_WEIGHTS['dxy_z_threshold']) * MACRO_WEIGHTS['dxy_penalty_per_sigma']
                macro_adjustment -= penalty
                regime_details.append(f"USD Spike (+{dxy_z:.1f}σ)")

            # =========================================================================
            # v11.2 Phase 2: VIX TERM STRUCTURE ANALYSIS
            # =========================================================================
            # Key insight from @alshfaw: 1.26 is historically BEARISH for SPY
            # Extreme contango (>1.20) = complacency trap, precedes corrections
            # Mild contango (1.10-1.20) = healthy low vol environment
            # Backwardation (<0.95) = fear/panic, imminent vol spike
            term_structure_ratio = vix3m_curr / vix_curr if vix_curr > 0 else 1.0
            ts_config = VIX_TERM_STRUCTURE_CONFIG

            if term_structure_ratio > ts_config['extreme_contango_threshold']:
                # EXTREME contango (>1.20) - COMPLACENCY WARNING
                # 1.26 brought markets down in Aug, Sep, Oct, Nov, Dec 2024
                term_structure = 'extreme_contango'
                macro_adjustment -= ts_config['extreme_contango_penalty']
                regime_details.append(f"⚠️ EXTREME Contango ({term_structure_ratio:.2f}) - Complacency Risk")
            elif term_structure_ratio > ts_config['mild_contango_threshold']:
                # Mild contango (1.10-1.20) - healthy low vol
                term_structure = 'contango'
                macro_adjustment += ts_config['mild_contango_bonus']
                regime_details.append(f"Contango ({term_structure_ratio:.2f})")
            elif term_structure_ratio < ts_config['backwardation_threshold']:
                # Backwardation (<0.95) - bearish/fear signal
                term_structure = 'backwardation'
                macro_adjustment -= ts_config['backwardation_penalty']
                regime_details.append(f"⚠️ Backwardation ({term_structure_ratio:.2f})")
            else:
                # Normal (0.95-1.10) - neutral
                term_structure = 'neutral'

            # =========================================================================
            # v11.2 Phase 2: VVIX DIVERGENCE DETECTION
            # =========================================================================
            # VVIX rising + VIX falling = vol spike incoming (bearish divergence)
            # VVIX falling + VIX rising = vol spike fading (bullish convergence)
            vvix_divergence = 'none'
            if vvix_series is not None and len(vvix_series) >= ts_config['vvix_divergence_lookback']:
                lookback = ts_config['vvix_divergence_lookback']
                vvix_change = (vvix_series.iloc[-1] - vvix_series.iloc[-lookback]) / vvix_series.iloc[-lookback]
                vix_change = (vix_series.iloc[-1] - vix_series.iloc[-lookback]) / vix_series.iloc[-lookback]

                # Bearish divergence: VVIX rising (>5%) while VIX falling (<-2%)
                if vvix_change > 0.05 and vix_change < -0.02:
                    vvix_divergence = 'bearish'
                    macro_adjustment -= ts_config['vvix_divergence_penalty']
                    regime_details.append(f"⚠️ VVIX Divergence (vol spike risk)")

                # Bullish convergence: VVIX falling (<-5%) while VIX rising (>2%)
                elif vvix_change < -0.05 and vix_change > 0.02:
                    vvix_divergence = 'bullish'
                    macro_adjustment += ts_config['vvix_convergence_bonus']
                    regime_details.append(f"VVIX Convergence (vol fading)")

                # Extreme VVIX warning (regardless of divergence)
                if vvix_curr > ts_config['vvix_high_threshold']:
                    regime_details.append(f"VVIX Elevated ({vvix_curr:.0f})")

            # 4. Define Regime Label
            if vix_z > 3.0: regime = "Crisis (Extreme Vol)"
            elif term_structure == 'backwardation' and vix_z > 1.0: regime = "Fear (Backwardation)"
            elif term_structure == 'extreme_contango': regime = "Complacency (Extreme Contango)"
            elif tnx_z > 3.0: regime = "Rate Shock"
            elif spy_curr > spy_series.iloc[-20]: regime = "Bull Trend"
            elif vix_curr > 25 and vix_z < 1.0: regime = "High Vol (New Normal)"
            else: regime = "Neutral"

            # Store Data (note: 'vix' key = VIX30D per @alshfaw terminology)
            self.macro_data = {
                'vix': vix_curr, 'vix_z': vix_z,  # vix = VIX30D (^VIX = 30-day implied vol)
                'tnx': tnx_curr, 'tnx_z': tnx_z,
                'dxy': dxy_curr, 'dxy_z': dxy_z,
                'spy_trend': spy_curr > spy_series.iloc[-20],
                'adjustment': macro_adjustment,
                'regime_details': regime_details,
                # v11.2 Phase 2: VIX Term Structure data (VIX3M/VIX30D ratio)
                'vix3m': vix3m_curr,
                'term_structure_ratio': term_structure_ratio,
                'term_structure': term_structure,
                'vvix': vvix_curr,
                'vvix_divergence': vvix_divergence
            }
            self.market_regime = regime

            # v11.2 Phase 2: Enhanced output with term structure
            # Note: VIX30D = ^VIX (standard VIX measures 30-day implied vol) per @alshfaw terminology
            print(f"  [MACRO] VIX30D: {vix_curr:.2f} ({vix_z:+.1f}σ) | TNX: {tnx_curr:.2f}% ({tnx_z:+.1f}σ) | DXY: {dxy_curr:.2f} ({dxy_z:+.1f}σ)")
            print(f"  [MACRO] VIX3M/VIX30D: {term_structure_ratio:.2f} | Structure: {term_structure.upper()} | VVIX: {vvix_curr:.0f}")
            print(f"  [MACRO] Regime: {regime} | Adjustment: {macro_adjustment:+.1f}")
            return regime

        except Exception as e:
            print(f"  [MACRO] Dynamic fetch failed: {e}. Using defaults.")
            self.macro_data = default_macro
            return "Neutral"

        # """
        # v10.7: Robust macro regime assessment with cooldown, individual fallback, and sanitization.
        # """
        # print("\n[TITAN] Assessing Macro Regime...")

        # # v10.7: Default values used if fetch fails completely
        # default_macro = {
        #     'vix': 20.0, 'tnx': 4.0, 'dxy': 100.0,
        #     'spy_trend': True, 'spy_return': 0.0,
        #     'adjustment': 0, 'regime_details': ['Data unavailable']
        # }

        # try:
        #     max_retries = 3
        #     vix = tnx = dxy = spy_current = spy_start = None

        #     # v10.7: Initial cooldown to let yfinance rate limiter reset
        #     time.sleep(2.0)

        #     for attempt in range(max_retries):
        #         try:
        #             # v10.7: Try batch download first (v10.8: with session)
        #             tickers = ['^VIX', '^TNX', 'DX-Y.NYB', 'SPY']
        #             data = yf.download(tickers, period="5d", progress=False, threads=False)

        #             # v10.7: Handle both single and multi-column return formats
        #             if 'Close' in data.columns or (isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0)):
        #                 close_data = data['Close'] if 'Close' in data.columns else data.xs('Close', axis=1, level=0)
        #             else:
        #                 close_data = data

        #             # Extract values with NaN checking
        #             vix = close_data['^VIX'].iloc[-1] if '^VIX' in close_data.columns else None
        #             tnx = close_data['^TNX'].iloc[-1] if '^TNX' in close_data.columns else None
        #             dxy = close_data['DX-Y.NYB'].iloc[-1] if 'DX-Y.NYB' in close_data.columns else None
        #             spy_current = close_data['SPY'].iloc[-1] if 'SPY' in close_data.columns else None
        #             spy_start = close_data['SPY'].iloc[0] if 'SPY' in close_data.columns else None

        #             # v10.7: Sanitize - convert to float and check for NaN
        #             try:
        #                 vix = float(vix) if vix is not None and not pd.isna(vix) else None
        #                 tnx = float(tnx) if tnx is not None and not pd.isna(tnx) else None
        #                 dxy = float(dxy) if dxy is not None and not pd.isna(dxy) else None
        #                 spy_current = float(spy_current) if spy_current is not None and not pd.isna(spy_current) else None
        #                 spy_start = float(spy_start) if spy_start is not None and not pd.isna(spy_start) else None
        #             except (TypeError, ValueError):
        #                 pass

        #             # Check for NaN values and retry if needed
        #             missing = []
        #             if vix is None: missing.append('VIX')
        #             if tnx is None: missing.append('TNX')
        #             if dxy is None: missing.append('DXY')
        #             if spy_current is None or spy_start is None: missing.append('SPY')

        #             if missing:
        #                 if attempt < max_retries - 1:
        #                     print(f"  [MACRO] Missing {', '.join(missing)}, retrying ({attempt + 1}/{max_retries})...")
        #                     time.sleep(2 ** (attempt + 1))  # Exponential backoff: 2, 4, 8 seconds
        #                     continue
        #                 else:
        #                     # v10.7: Try individual fetches as last resort
        #                     print(f"  [MACRO] Batch failed, trying individual fetches...")
        #                     time.sleep(2.0)

        #                     if vix is None:
        #                         try:
        #                             vix_data = yf.download('^VIX', period='5d', progress=False)
        #                             vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else default_macro['vix']
        #                         except: vix = default_macro['vix']
        #                         time.sleep(1.0)

        #                     if tnx is None:
        #                         try:
        #                             tnx_data = yf.download('^TNX', period='5d', progress=False)
        #                             tnx = float(tnx_data['Close'].iloc[-1]) if not tnx_data.empty else default_macro['tnx']
        #                         except: tnx = default_macro['tnx']
        #                         time.sleep(1.0)

        #                     if dxy is None:
        #                         try:
        #                             dxy_data = yf.download('DX-Y.NYB', period='5d', progress=False)
        #                             dxy = float(dxy_data['Close'].iloc[-1]) if not dxy_data.empty else default_macro['dxy']
        #                         except: dxy = default_macro['dxy']
        #                         time.sleep(1.0)

        #                     if spy_current is None or spy_start is None:
        #                         try:
        #                             spy_data = yf.download('SPY', period='5d', progress=False)
        #                             if not spy_data.empty:
        #                                 spy_current = float(spy_data['Close'].iloc[-1])
        #                                 spy_start = float(spy_data['Close'].iloc[0])
        #                         except:
        #                             spy_current = 500.0
        #                             spy_start = 500.0

        #             break  # Exit retry loop

        #         except Exception as retry_error:
        #             if attempt < max_retries - 1:
        #                 print(f"  [MACRO] Fetch attempt {attempt + 1} failed: {str(retry_error)[:50]}")
        #                 time.sleep(2 ** (attempt + 1))
        #                 continue
        #             else:
        #                 raise retry_error

        #     # v10.7: Final sanitization - ensure we have valid values
        #     vix = vix if vix is not None and 5 < vix < 100 else default_macro['vix']
        #     tnx = tnx if tnx is not None and 0 < tnx < 20 else default_macro['tnx']
        #     dxy = dxy if dxy is not None and 80 < dxy < 130 else default_macro['dxy']
        #     spy_current = spy_current if spy_current is not None else 500.0
        #     spy_start = spy_start if spy_start is not None else 500.0

        #     spy_trend = spy_current > spy_start
        #     spy_return = (spy_current - spy_start) / spy_start if spy_start > 0 else 0

        #     # Store raw macro values for score weighting
        #     self.macro_data = {
        #         'vix': vix,
        #         'tnx': tnx,
        #         'dxy': dxy,
        #         'spy_trend': spy_trend,
        #         'spy_return': spy_return
        #     }

        #     # Calculate regime based on successfully fetched data
        #     regime = "Neutral"
        #     regime_details = []

        #     if vix > 30:
        #         regime = "Bear Volatility"
        #         regime_details.append(f"High fear (VIX {vix:.1f})")
        #     elif vix > 20 and not spy_trend:
        #         regime = "Correction"
        #         regime_details.append(f"Elevated volatility + weakness")
        #     elif tnx > 4.5 and dxy > 105:
        #         regime = "Rates Pressure"
        #         regime_details.append(f"Yields pressuring equities (TNX {tnx:.2f}%)")
        #     elif spy_trend and vix < 20:
        #         regime = "Bull Trend"
        #         regime_details.append(f"Risk-on environment")
        #     else:
        #         regime_details.append("Mixed signals")

        #     # Calculate macro adjustment score (used to weight individual picks)
        #     macro_adjustment = 0
        #     if vix > MACRO_WEIGHTS['vix_penalty_threshold']:
        #         macro_adjustment -= (vix - MACRO_WEIGHTS['vix_penalty_threshold']) * MACRO_WEIGHTS['vix_penalty_per_point']
        #     if tnx > MACRO_WEIGHTS['tnx_penalty_threshold']:
        #         macro_adjustment -= (tnx - MACRO_WEIGHTS['tnx_penalty_threshold']) * MACRO_WEIGHTS['tnx_penalty_per_point']
        #     if dxy > MACRO_WEIGHTS['dxy_strength_threshold']:
        #         macro_adjustment -= (dxy - MACRO_WEIGHTS['dxy_strength_threshold']) * MACRO_WEIGHTS['dxy_penalty_per_point']

        #     self.macro_data['adjustment'] = macro_adjustment
        #     self.macro_data['regime_details'] = regime_details

        #     print(f"  [MACRO] VIX: {vix:.2f} | TNX: {tnx:.2f}% | DXY: {dxy:.2f} | Regime: {regime}")
        #     print(f"  [MACRO] Score Adjustment: {macro_adjustment:+.1f} points")
        #     self.market_regime = regime
        #     return regime

        # except Exception as e:
        #     print(f"  [MACRO] Data fetch failed ({str(e)[:50]}). Using defaults.")
        #     self.macro_data = default_macro.copy()
        #     self.market_regime = "Neutral"
        #     return "Neutral"

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

    def extract_strike_gamma(self, big_filepath):
        """
        Lightweight extraction of strike-level gamma data for GEX wall analysis.
        Called separately when cached aggregated data is used.
        """
        print(f"  [GEX] Extracting strike-level gamma from raw file...")
        chunk_size = 200000
        strike_chunks = []

        try:
            for chunk in pd.read_csv(big_filepath, chunksize=chunk_size, low_memory=False):
                chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_')

                # Quick filter for significant options
                if 'gamma' not in chunk.columns or 'strike' not in chunk.columns:
                    continue

                mask = chunk['gamma'].abs() > 0.001
                chunk = chunk[mask].copy()
                if chunk.empty:
                    continue

                if 'underlying_symbol' in chunk.columns:
                    chunk['ticker'] = chunk['underlying_symbol'].apply(self.normalize_ticker)

                # Calculate net gamma
                chunk['weight'] = 0.0
                if 'side' in chunk.columns:
                    chunk.loc[chunk['side'].str.upper().isin(['ASK', 'A', 'BUY']), 'weight'] = 1.0
                    chunk.loc[chunk['side'].str.upper().isin(['BID', 'B', 'SELL']), 'weight'] = -0.5

                mult = chunk['size'] if 'size' in chunk.columns else 100
                chunk['net_gamma'] = chunk['gamma'] * mult * chunk['weight']

                # Keep only significant gamma entries
                strike_data = chunk[['ticker', 'strike', 'net_gamma']].copy()
                strike_data = strike_data[strike_data['net_gamma'].abs() > 50]  # Lower threshold
                if not strike_data.empty:
                    strike_chunks.append(strike_data)

            if strike_chunks:
                strike_df = pd.concat(strike_chunks, ignore_index=True)
                # Aggregate by ticker+strike
                strike_agg = strike_df.groupby(['ticker', 'strike'])['net_gamma'].sum().reset_index()
                # Store as dict: ticker -> {strike: gamma}
                for ticker in strike_agg['ticker'].unique():
                    ticker_strikes = strike_agg[strike_agg['ticker'] == ticker]
                    self.strike_gamma_data[ticker] = dict(zip(ticker_strikes['strike'], ticker_strikes['net_gamma']))
                print(f"  [GEX] Extracted strike-level gamma for {len(self.strike_gamma_data)} tickers")
            else:
                print(f"  [GEX] No strike data found in file (check if 'strike' column exists)")

        except Exception as e:
            print(f"  [!] Strike extraction error: {e}")

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

    # --- HIVE MIND (ENSEMBLE) - See engine/transformer.py for full implementation ---
    def train_run_transformer(self):
        """Train ensemble of transformers (Hive Mind). See engine/transformer.py (~150 lines)."""
        nn_df = _train_hive_mind(self.history_mgr, self.nn_scaler, device, num_models=5)
        if nn_df is not None and not self.full_df.empty:
            self.full_df = pd.merge(self.full_df, nn_df, on='ticker', how='left').fillna(0)

    # --- RESTORED SECTOR FETCH ---
    def fetch_sector_history(self):
        print("  [CONTEXT] Fetching Sector ETF History...")
        try:
            etfs = list(SECTOR_MAP.values()) + ['SPY']
            # Fetch 1-month data for relative performance
            data = yf.download(etfs, period="1mo", progress=False)['Close']
            if isinstance(data, pd.Series): data = data.to_frame()
            for etf in etfs:
                if etf in data.columns:
                    series = data[etf].dropna()
                    if len(series) > 1:
                        self.sector_data[etf] = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
                    else: self.sector_data[etf] = 0.0

            # v11.5.4: Fetch 6-month data for sector relative performance filter
            # Used to compare stock performance vs sector performance
            # True phoenix = stock outperforming sector (individual alpha)
            # Sector beta = stock just matching sector (riding the wave)
            data_6m = yf.download(etfs, period="6mo", progress=False)['Close']
            if isinstance(data_6m, pd.Series): data_6m = data_6m.to_frame()
            self.sector_6m_returns = {}
            for etf in etfs:
                if etf in data_6m.columns:
                    series = data_6m[etf].dropna()
                    if len(series) > 20:
                        start_price = series.iloc[0]
                        current_price = series.iloc[-1]
                        return_6m = (current_price - start_price) / start_price if start_price > 0 else 0.0
                        self.sector_6m_returns[etf] = float(return_6m)
        except Exception as e:
            pass

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

    # --- PHASE 9: PATTERN DETECTION METHODS (v12 Modular - Thin Wrappers) ---
    # Full implementations moved to engine/patterns.py (~800 lines saved)

    def detect_bull_flag(self, ticker, history_df, debug=False):
        """Detect bull flag pattern. See engine/patterns.py for full implementation."""
        return _detect_bull_flag(history_df)

    def find_gex_walls(self, ticker, current_price, bot_df=None):
        """Find GEX walls. See engine/patterns.py for full implementation."""
        return _find_gex_walls(ticker, current_price, self.strike_gamma_data, self.dp_support_levels, bot_df)

    def detect_downtrend_reversal(self, ticker, history_df):
        """Detect reversal setup. See engine/patterns.py for full implementation."""
        return _detect_downtrend_reversal(ticker, history_df, self.dp_support_levels)

    def calculate_flow_factor(self, ticker, volume_ratio=1.0):
        """Calculate flow factor. See engine/patterns.py for full implementation."""
        return _calculate_flow_factor(ticker, volume_ratio, self.full_df, getattr(self, 'signature_prints', None))

    # --- v11.0 SMART GATEKEEPER ---
    def apply_smart_gatekeeper(self, df):
        """Smart gatekeeper filter. See engine/patterns.py for full implementation."""
        return _apply_smart_gatekeeper(df)

    # --- v11.0 SOLIDITY SCORE ---
    def calculate_solidity_score(self, ticker, history_df):
        """Calculate solidity score. See engine/patterns.py for full implementation."""
        return _calculate_solidity_score(ticker, history_df, self.full_df, self.dp_support_levels)

    def detect_phoenix_reversal(self, ticker, history_df):
        """Detect phoenix reversal. See engine/patterns.py for full implementation (~400 lines)."""
        return _detect_phoenix_reversal(ticker, history_df, self.full_df, self.dp_support_levels, getattr(self, 'signature_prints', None))

    def detect_cup_and_handle(self, ticker, history_df):
        """Detect cup-and-handle pattern. See engine/patterns.py for full implementation."""
        return _detect_cup_and_handle(history_df)

    def detect_double_bottom(self, ticker, history_df):
        """Detect double-bottom pattern. See engine/patterns.py for full implementation."""
        return _detect_double_bottom(history_df)

    # --- PHASE 10: POSITION SIZING (Kelly Criterion) ---
    def calculate_position_size(self, row, portfolio_value=100000):
        """
        Calculate position size using Modified Kelly Criterion.

        Kelly Formula: f* = (p*b - q) / b
        Where: p = win probability, q = loss probability (1-p), b = win/loss ratio

        Simplified as: Position = (Edge * P(win) - P(loss)) / Vol_Risk
        Capped at quarter-Kelly for safety.

        Args:
            row: DataFrame row with ticker data (nn_score, atr, current_price, etc.)
            portfolio_value: Total portfolio value for sizing

        Returns:
            dict with position_size, position_pct, risk_tier, kelly_raw, etc.
        """
        result = {
            'position_size': 0,
            'position_pct': 0.0,
            'shares': 0,
            'risk_tier': 'MEDIUM',
            'vol_bucket': 'medium',
            'kelly_raw': 0.0,
            'kelly_capped': 0.0,
            'sizing_explanation': ''
        }

        try:
            # Extract key metrics
            nn_score = row.get('nn_score', 50)
            current_price = row.get('current_price', 0)
            atr = row.get('atr', 0)

            if current_price <= 0:
                result['sizing_explanation'] = 'Invalid price'
                return result

            # Calculate volatility ratio (ATR/Price)
            vol_ratio = atr / current_price if current_price > 0 else 0.10

            # Determine vol bucket
            if vol_ratio < RISK_TIER_MATRIX['vol_thresholds']['low']:
                vol_bucket = '<5%'
                result['vol_bucket'] = 'low'
            elif vol_ratio < RISK_TIER_MATRIX['vol_thresholds']['medium']:
                vol_bucket = '5-6%'
                result['vol_bucket'] = 'medium'
            else:
                vol_bucket = '>6%'
                result['vol_bucket'] = 'high'

            # Determine score tier and get multiplier
            multiplier = 0.20  # Default for low scores
            for (low, high), vol_dict in RISK_TIER_MATRIX['score_tiers'].items():
                if low <= nn_score <= high:
                    multiplier = vol_dict.get(vol_bucket, 0.20)
                    if nn_score >= 95:
                        result['risk_tier'] = 'HIGH_CONVICTION'
                    elif nn_score >= 85:
                        result['risk_tier'] = 'MEDIUM_HIGH'
                    else:
                        result['risk_tier'] = 'MEDIUM'
                    break

            # Estimate edge based on model confidence
            # nn_score of 90 = 0.15 edge, 80 = 0.10 edge, 70 = 0.05 edge
            estimated_edge = max(0.02, (nn_score - 50) / 400)  # 0.02 to 0.125

            # Estimate win rate from score (conservative)
            win_rate = min(0.65, 0.50 + (nn_score - 50) / 200)  # 0.50 to 0.65

            # Kelly calculation: f* = (p * b - q) / b, where b = edge / (1 - edge)
            # Simplified: f* = (win_rate * (1 + edge) - (1 - win_rate)) / (1 + edge)
            loss_rate = 1 - win_rate
            if vol_ratio > 0:
                kelly_raw = (estimated_edge * win_rate - loss_rate) / vol_ratio
            else:
                kelly_raw = 0

            # Cap at quarter-Kelly for safety
            kelly_fraction = POSITION_SIZING_CONFIG['kelly_fraction']
            kelly_capped = max(0, kelly_raw * kelly_fraction)

            # Apply tier multiplier
            position_pct = kelly_capped * multiplier

            # Enforce min/max bounds
            min_pct = POSITION_SIZING_CONFIG['min_position_pct']
            max_pct = POSITION_SIZING_CONFIG['max_position_pct']
            position_pct = max(min_pct, min(max_pct, position_pct))

            # Calculate dollar amount and shares
            position_size = portfolio_value * position_pct
            shares = int(position_size / current_price) if current_price > 0 else 0

            # Populate result
            result['position_size'] = round(position_size, 2)
            result['position_pct'] = round(position_pct * 100, 2)  # As percentage
            result['shares'] = shares
            result['kelly_raw'] = round(kelly_raw, 4)
            result['kelly_capped'] = round(kelly_capped, 4)
            result['sizing_explanation'] = (
                f"{result['risk_tier']} | Vol:{result['vol_bucket']} | "
                f"Kelly:{kelly_raw:.1%}→{position_pct:.1%} | "
                f"${position_size:,.0f} ({shares} shares)"
            )

        except Exception as e:
            result['sizing_explanation'] = f'Sizing error: {str(e)[:30]}'

        return result

    def get_vol_bucket_label(self, vol_ratio):
        """Helper to get human-readable vol bucket label."""
        if vol_ratio < 0.05:
            return 'LOW (<5%)'
        elif vol_ratio < 0.06:
            return 'MEDIUM (5-6%)'
        else:
            return 'HIGH (>6%)'

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

            # Phoenix
            if patterns.get('phoenix', {}).get('is_phoenix'):
                reasons.append(patterns['phoenix']['explanation'])

            # Cup-and-Handle
            if patterns.get('cup_handle', {}).get('is_cup_handle'):
                reasons.append(patterns['cup_handle']['explanation'])

            # Double Bottom
            if patterns.get('double_bottom', {}).get('is_double_bottom'):
                reasons.append(patterns['double_bottom']['explanation'])

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
            # ALSO extract strike-level data from raw file for GEX analysis
            if file_map.get('bot_big') and os.path.exists(file_map.get('bot_big')):
                self.extract_strike_gamma(file_map.get('bot_big'))
        elif file_map.get('bot_big') and os.path.exists(file_map.get('bot_big')):
            df_bot = self.optimize_large_dataset(file_map.get('bot_big'), date_stamp=None)
        elif not df_screener.empty:
            df_bot = df_screener.copy()
            if 'net_call_premium' in df_bot.columns:
                df_bot['screener_flow'] = df_bot['net_call_premium'] - df_bot['net_put_premium']
                df_bot['net_gamma'] = df_bot['screener_flow'] / 100.0

        # --- PATTERN VALIDATION SUITE (v10.6: Uses module-level VALIDATION_SUITE) ---
        # Force-include known institutional phoenix patterns to validate detection
        # v10.6 FIX: Now forces validation tickers into top 75 (see predict() method)

        if ENABLE_VALIDATION_MODE:
            # Combine all test tickers from module-level VALIDATION_SUITE
            test_tickers = (
                VALIDATION_SUITE['institutional_phoenix'] +
                VALIDATION_SUITE['speculative_phoenix']
            )

            if not df_bot.empty and 'ticker' in df_bot.columns and test_tickers:
                print(f"\n  [VALIDATION MODE] Adding {len(test_tickers)} tickers to candidate pool...")
                for test_ticker in test_tickers:
                    if test_ticker not in df_bot['ticker'].values:
                        # Add minimal row for forced ticker (will be force-included in top 75 later)
                        force_row = {
                            'ticker': test_ticker,
                            'net_gamma': 0.0,  # Low score OK - we force into top 75 anyway
                            'net_flow': 0.0,
                            'dp_total': 0.0
                        }
                        df_bot = pd.concat([df_bot, pd.DataFrame([force_row])], ignore_index=True)
                        print(f"    → Added {test_ticker} (will be force-included in pattern analysis)")
        # --- END VALIDATION SUITE ---

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
        high = history_df['High']
        low = history_df['Low']
        volume = history_df['Volume']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]
        if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]

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

        # --- ORDER FLOW IMBALANCE FEATURES (v12) ---
        # Close Location Value: where price closed in day's range (-1 to +1)
        # +1 = closed at high (buying pressure), -1 = closed at low (selling pressure)
        hl_range = high - low
        clv = ((close - low) - (high - close)) / (hl_range + 1e-9)
        clv = clv.clip(-1, 1)  # Bound to [-1, 1]

        # Money Flow Volume: CLV-weighted volume
        mf_volume = clv * volume

        # Chaikin Money Flow (20-day): normalized accumulation/distribution
        cmf_20 = mf_volume.rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)

        # On-Balance Volume: cumulative volume based on price direction
        obv = (volume * np.sign(delta)).fillna(0).cumsum()
        # OBV slope (momentum of flow): 10-day rate of change normalized
        obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / (abs(obv.iloc[-10]) + 1e-9) if len(obv) > 10 else 0.0

        # Volume-Weighted Average Price distance (intraday proxy)
        vwap_proxy = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        vwap_distance = (close.iloc[-1] - vwap_proxy.iloc[-1]) / (vwap_proxy.iloc[-1] + 1e-9)

        # --- FRACTIONAL DIFFERENTIATION (v12) ---
        # López de Prado's method: make series stationary while preserving memory
        # d=0.3-0.5 balances stationarity with information retention
        # Reference: "Advances in Financial Machine Learning" Ch. 5
        def frac_diff_weights(d, threshold=1e-4, max_k=100):
            """Generate fractional differentiation weights until threshold."""
            weights = [1.0]
            for k in range(1, max_k):
                w = weights[-1] * (d - k + 1) / k
                if abs(w) < threshold:
                    break
                weights.append(w)
            return np.array(weights[::-1])  # Reverse for convolution

        def apply_frac_diff(series, d=0.4):
            """Apply fractional differentiation to a series."""
            weights = frac_diff_weights(d)
            width = len(weights)
            if len(series) < width:
                return np.nan
            # Use log prices for fractional differentiation
            log_series = np.log(series.replace(0, 1e-9))
            # Convolve with weights (dot product of last 'width' values)
            result = log_series.rolling(width).apply(lambda x: np.dot(x, weights), raw=True)
            return result

        # Fractionally differentiated log close (d=0.4 typical for stocks)
        frac_diff = apply_frac_diff(close, d=0.4)
        frac_diff_close = float(frac_diff.iloc[-1]) if not pd.isna(frac_diff.iloc[-1]) else 0.0

        # --- VPIN FLOW TOXICITY (v12) ---
        # Volume-synchronized Probability of Informed Trading
        # Measures order flow imbalance to detect informed trading activity
        # Reference: Easley, López de Prado, O'Hara (2012) "Flow Toxicity and Liquidity"
        open_price = history_df['Open']
        if isinstance(open_price, pd.DataFrame):
            open_price = open_price.iloc[:, 0]

        # Bulk Volume Classification: classify daily volume as buy/sell
        # Using close-open direction (simplified tick rule for daily data)
        price_direction = np.sign(close - open_price)  # +1 buy, -1 sell, 0 neutral
        buy_volume = volume * ((price_direction + 1) / 2)  # Scale: 0 to 1
        sell_volume = volume * ((1 - price_direction) / 2)  # Scale: 0 to 1

        # Order Imbalance per period
        order_imbalance = (buy_volume - sell_volume) / (volume + 1e-9)

        # VPIN: Rolling average of absolute order imbalance (20-day window)
        # Higher VPIN = more informed trading (potential volatility/moves)
        vpin_20 = order_imbalance.abs().rolling(20).mean()
        vpin = float(vpin_20.iloc[-1]) if not pd.isna(vpin_20.iloc[-1]) else 0.0

        # VPIN velocity: rate of change in VPIN (rising VPIN = increasing informed activity)
        vpin_prev = float(vpin_20.iloc[-5]) if len(vpin_20) > 5 and not pd.isna(vpin_20.iloc[-5]) else vpin
        vpin_velocity = (vpin - vpin_prev) / (vpin_prev + 1e-9)

        # --- SMART MONEY CONCEPTS (v12) ---
        # ICT-style patterns for institutional order flow detection
        from engine.patterns import detect_smc_patterns
        smc = detect_smc_patterns(history_df, lookback=30)
        smc_bullish = smc['smc_bullish_score']
        smc_bearish = smc['smc_bearish_score']
        # Net SMC signal: positive = bullish structure, negative = bearish structure
        smc_net = smc_bullish - smc_bearish

        return {
            'rsi': float(rsi.iloc[-1]),
            'trend_score': float(trend_score.iloc[-1]),
            'volatility': float(volatility.iloc[-1]),
            'flag_score': 0.0,
            'divergence_score': float(div_score),
            'sma_alignment': 1 if sma20.iloc[-1] > sma50.iloc[-1] else 0,
            'lagged_return_5d': float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) if len(close) > 6 else 0.0,
            'current_price': float(close.iloc[-1]),
            'dist_sma50': float(dist_sma50),
            # Order Flow Imbalance features (v12)
            'clv': float(clv.iloc[-1]),  # Close location value (-1 to +1)
            'cmf_20': float(cmf_20.iloc[-1]),  # Chaikin Money Flow 20-day
            'obv_slope': float(obv_slope),  # OBV momentum
            'vwap_distance': float(vwap_distance),  # Distance from VWAP proxy
            # Fractional Differentiation (v12)
            'frac_diff_close': frac_diff_close,  # Stationary price with memory
            # VPIN Flow Toxicity (v12)
            'vpin': vpin,  # Volume-synchronized Prob of Informed Trading (0-1)
            'vpin_velocity': vpin_velocity,  # Rate of change in VPIN
            # Smart Money Concepts (v12)
            'smc_bullish': smc_bullish,  # CHoCH + Order Blocks + FVG bullish score
            'smc_bearish': smc_bearish,  # CHoCH + Order Blocks + FVG bearish score
            'smc_net': smc_net,  # Net SMC signal (-1 to +1)
        }

    def enrich_market_data(self, flow_df):
        print("\n[2/4] Enriching with Price History (Deep Mode via Alpaca)...")
        if flow_df.empty: return self.full_df

        # Fetch Sector Data
        try:
            self.fetch_sector_history()
        except: pass

        cached_data = {}
        if os.path.exists(self.price_cache_file):
            try:
                df_cache = pd.read_csv(self.price_cache_file)
                cached_data = df_cache.set_index('ticker').to_dict('index')
            except: pass

        if 'ticker' not in flow_df.columns: return flow_df
        tickers = flow_df['ticker'].unique().tolist()

        # Filter: only fetch what we don't have in cache
        to_fetch = [t for t in tickers if t not in cached_data and len(str(t)) < 8 and str(t) != 'nan']

        # FIX: Prioritize validation tickers to ensure they're always fetched
        # Even if we hit the 3000 limit, these tickers must be included
        # Also add validation tickers even if not in flow_df (ensures price data exists)
        if ENABLE_VALIDATION_MODE:
            priority_tickers = (
                VALIDATION_SUITE.get('institutional_phoenix', []) +
                VALIDATION_SUITE.get('speculative_phoenix', [])
            )
            # Add validation tickers that aren't in flow_df but need price data
            missing_from_flow = [t for t in priority_tickers if t not in tickers and t not in cached_data]
            if missing_from_flow:
                print(f"  [FETCH] Adding validation tickers missing from flow data: {missing_from_flow}")
                to_fetch = missing_from_flow + to_fetch

            # Move priority tickers to front of list
            priority_in_fetch = [t for t in priority_tickers if t in to_fetch]
            other_tickers = [t for t in to_fetch if t not in priority_tickers]
            to_fetch = priority_in_fetch + other_tickers
            if priority_in_fetch:
                print(f"  [FETCH] Prioritized {len(priority_in_fetch)} validation tickers: {priority_in_fetch}")

        # Limit downloads for performance
        max_fetch = PERFORMANCE_CONFIG.get('max_tickers_to_fetch', 3000)
        if len(to_fetch) > max_fetch:
            print(f"  [FETCH] Limiting from {len(to_fetch)} to {max_fetch} tickers for performance")
            to_fetch = to_fetch[:max_fetch]

        if to_fetch:
            print(f"  [FETCH] Downloading {len(to_fetch)} tickers via Alpaca...")

            # Use global Alpaca helper to fetch 3 months of data
            start_date_3mo = datetime.now() - timedelta(days=90)
            fetched_data = fetch_alpaca_batch(to_fetch, start_date=start_date_3mo)

            for ticker, hist in fetched_data.items():
                try:
                    metrics = self.calculate_technicals(hist)
                    if metrics:
                        cached_data[ticker] = metrics
                except Exception as e:
                    pass

            # Save updated cache
            try:
                pd.DataFrame.from_dict(cached_data, orient='index').reset_index().rename(columns={'index':'ticker'}).to_csv(self.price_cache_file, index=False)
                print(f"  [FETCH] Updated cache with {len(fetched_data)} new tickers.")
            except: pass

        self.market_breadth = self.calculate_market_breadth(cached_data)
        tech_df = pd.DataFrame.from_dict(cached_data, orient='index').reset_index().rename(columns={'index':'ticker'})

        if tech_df.empty:
            self.full_df = flow_df.copy()
            for c in ['trend_score', 'rsi']: self.full_df[c] = 0
            return self.full_df

        final_df = pd.merge(flow_df, tech_df, on='ticker', how='left')
        self.full_df = final_df[final_df['rsi'].notna()].fillna(0)

        # FIX: Ensure validation tickers are in full_df even if missing from flow data
        # This handles cases where LULU isn't in the daily bot-eod-report but we have price data
        if ENABLE_VALIDATION_MODE:
            priority_tickers = (
                VALIDATION_SUITE.get('institutional_phoenix', []) +
                VALIDATION_SUITE.get('speculative_phoenix', [])
            )
            for ticker in priority_tickers:
                if ticker not in self.full_df['ticker'].values and ticker in tech_df['ticker'].values:
                    # Create synthetic row with price data only (no flow data)
                    ticker_tech = tech_df[tech_df['ticker'] == ticker].iloc[0].to_dict()
                    # Add default flow columns
                    for col in self.full_df.columns:
                        if col not in ticker_tech:
                            ticker_tech[col] = 0
                    ticker_tech['ticker'] = ticker
                    self.full_df = pd.concat([self.full_df, pd.DataFrame([ticker_tech])], ignore_index=True)
                    print(f"  [VALIDATION] Added {ticker} to candidate pool (price data only, no flow data)")

        return self.full_df

    def train_model(self, force_retrain=False):
        """Train diverse ensemble. See engine/ml.py for full implementation (~500 lines)."""
        if self.full_df.empty:
            return

        # Build cache paths dict for module function
        cache_paths = {
            'catboost': self.catboost_path,
            'tabnet': self.tabnet_path,
            'tcn': self.tcn_path,
            'elasticnet': self.elasticnet_path,
            'meta_learner': self.meta_learner_path
        }

        # Call extracted training function
        result = _train_ensemble(
            full_df=self.full_df,
            history_mgr=self.history_mgr,
            imputer=self.imputer,
            scaler=self.scaler,
            features_list=self.features_list,
            cache_paths=cache_paths,
            market_regime=self.market_regime,
            force_retrain=force_retrain
        )

        # Update instance state with trained models
        if result:
            self.catboost_model = result['catboost_model']
            self.tabnet_model = result['tabnet_model']
            self.tcn_model = result['tcn_model']
            self.tcn_n_features = result['tcn_n_features']
            self.tcn_seq_len = result['tcn_seq_len']
            self.elasticnet_model = result['elasticnet_model']
            self.meta_learner = result['meta_learner']
            self.imputer = result['imputer']
            self.scaler = result['scaler']
            self.features_list = result['features_list']
            self.model_trained = result['model_trained']

    def predict(self):
        if self.full_df.empty: return None
        print("\n[4/4] Generating Predictions with Pattern Intelligence...")
        df = self.full_df.copy()

        # --- BASE SCORE FROM ENSEMBLE STACK ---
        if self.model_trained:
            # v12: Filter features_list to only include columns that exist in full_df
            # This handles cases where cached model was trained with different features
            available_features = [f for f in self.features_list if f in self.full_df.columns]
            missing_features = [f for f in self.features_list if f not in self.full_df.columns]
            if missing_features:
                print(f"  [WARN] Missing {len(missing_features)} features from cached model: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                print(f"  [WARN] Using {len(available_features)}/{len(self.features_list)} available features. Consider force_retrain=True")

            if not available_features:
                print("  [ERROR] No features available for prediction, using raw_score = 0.5")
                df['raw_score'] = 0.5
                self.full_df = df
            else:
                X = self.full_df[available_features]
                X_clean = self.imputer.transform(X)
                X_scaled = self.scaler.transform(X_clean)

                # Get predictions from diverse ensemble (CatBoost + TabNet + TCN + ElasticNet)
                preds_list = [self.catboost_model.predict_proba(X_scaled)[:, 1]]  # CatBoost always available

                if self.tabnet_model is not None:
                    preds_list.append(self.tabnet_model.predict_proba(X_scaled)[:, 1])

                # TCN is NOT included in meta-learner ensemble
                # It uses different features (6 temporal vs 15 tabular) and was trained separately
                # The meta-learner expects [CB, TN, EN] predictions only
                # TCN could be used as a standalone signal boost in the future if needed

                if self.elasticnet_model is not None:
                    preds_list.append(self.elasticnet_model.predict_proba(X_scaled)[:, 1])

                # Stack predictions for meta-learner
                X_meta = np.column_stack(preds_list)

                # Get final ensemble prediction from meta-learner
                probs = self.meta_learner.predict_proba(X_meta)[:, 1]
                df['raw_score'] = probs
        else:
            df['raw_score'] = 0.5

        # --- v11.0 DUAL-RANKING ARCHITECTURE ---
        # Create separate scores for Alpha Momentum vs Phoenix Reversals
        # These are NOT the same as the old trend/ambush scores - they use different formulas
        print("  [v11.2] Initializing Dual-Ranking Architecture...")

        # --- v12: REGIME-ADAPTIVE WEIGHT ADJUSTMENTS ---
        # Get regime-specific weight multipliers
        regime_adjustments = REGIME_WEIGHT_ADJUSTMENTS.get(self.market_regime, REGIME_WEIGHT_ADJUSTMENTS['Neutral'])
        alpha_regime_adj = regime_adjustments.get('alpha', {})
        phoenix_regime_adj = regime_adjustments.get('phoenix', {})
        regime_score_adj = regime_adjustments.get('score_adjustment', 0)
        feature_boost = regime_adjustments.get('feature_boost', [])

        if alpha_regime_adj or phoenix_regime_adj:
            print(f"  [v12] Regime-Adaptive Mode: {self.market_regime}")
            if feature_boost:
                print(f"  [v12] Boosted features: {', '.join(feature_boost)}")

        # --- ALPHA MOMENTUM SCORE (for continuation/trending plays) ---
        # Formula: Heavy weight on ML prediction + trend + volume momentum
        alpha_config = DUAL_RANKING_CONFIG['alpha_momentum'].copy()
        # Apply regime adjustments to alpha weights
        for key, multiplier in alpha_regime_adj.items():
            if key in alpha_config:
                alpha_config[key] = alpha_config[key] * multiplier

        # Base from ML ensemble (25% weight)
        df['alpha_momentum_score'] = df['raw_score'] * 100 * alpha_config['weight_ml']

        # Trend component (30% weight) - based on RSI and price action
        if 'rsi' in df.columns:
            # RSI 50-70 is optimal for momentum (not overbought, not oversold)
            rsi_score = df['rsi'].apply(lambda x: 100 if 50 <= x <= 70 else max(0, 100 - abs(x - 60) * 2))
            df['alpha_momentum_score'] += rsi_score * alpha_config['weight_trend']

        # Neural network component (15% weight)
        if 'nn_score' in df.columns:
            df['alpha_momentum_score'] += df['nn_score'] * alpha_config['weight_neural']

        # Volume momentum component (15% weight)
        if 'gamma_velocity' in df.columns:
            vol_momentum = df['gamma_velocity'].clip(0, 100)
            df['alpha_momentum_score'] += vol_momentum * alpha_config['weight_volume']

        # --- PHOENIX REVERSAL SCORE (for base breakout/reversal plays) ---
        # Formula: Heavy weight on solidity + duration + institutional flow, LESS on ML
        phoenix_config = DUAL_RANKING_CONFIG['phoenix_reversal'].copy()
        # Apply regime adjustments to phoenix weights
        for key, multiplier in phoenix_regime_adj.items():
            if key in phoenix_config:
                phoenix_config[key] = phoenix_config[key] * multiplier

        # Base from ML ensemble (12% weight - LOWER than alpha, ML is momentum-biased)
        df['phoenix_reversal_score'] = df['raw_score'] * 100 * phoenix_config['weight_ml']

        # Institutional flow component (20% weight)
        if 'dp_total' in df.columns:
            # Scale DP total to 0-100 (log scale for mega-prints)
            import math
            dp_score = df['dp_total'].apply(lambda x: min(100, math.log10(max(x, 1)) * 10) if x > 0 else 0)
            df['phoenix_reversal_score'] += dp_score * phoenix_config['weight_flow']

        # Net gamma contribution to institutional flow
        if 'net_gamma' in df.columns:
            gamma_contribution = df['net_gamma'].apply(lambda x: min(50, abs(x) / 10000))
            df['phoenix_reversal_score'] += gamma_contribution * (phoenix_config['weight_flow'] / 2)

        # Keep old trend/ambush scores for backwards compatibility
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
        # Phoenix doesn't penalize low flow as heavily (reversals start with low activity)
        df.loc[~flow_support, 'phoenix_reversal_score'] -= 10

        # --- NEURAL NETWORK BOOST ---
        if 'nn_score' in df.columns:
            boost = (df['nn_score'] - 50) * 0.5
            df['trend_score_val'] += boost
            df['ambush_score_val'] += boost

        # --- GAMMA VELOCITY BOOST ---
        if 'gamma_velocity' in df.columns:
            df.loc[df['gamma_velocity'] > 50, 'trend_score_val'] += 5
            df.loc[df['gamma_velocity'] > 50, 'alpha_momentum_score'] += 5

        # --- MACRO ADJUSTMENT (Phase 9) ---
        macro_adj = self.macro_data.get('adjustment', 0)
        if macro_adj != 0:
            print(f"  [MACRO] Applying {macro_adj:+.1f} point adjustment to all scores")
            df['trend_score_val'] += macro_adj
            df['ambush_score_val'] += macro_adj
            df['alpha_momentum_score'] += macro_adj
            df['phoenix_reversal_score'] += macro_adj

        # --- v12: REGIME FEATURE BOOST ---
        # Apply bonus points for regime-relevant features
        if feature_boost:
            boost_bonus = 0.0
            for feat in feature_boost:
                if feat in df.columns:
                    # Normalize feature to 0-10 points and add to scores
                    feat_vals = df[feat].fillna(0)
                    # For features in [-1, 1] range, scale to [0, 5]
                    if feat in ['clv', 'cmf_20', 'smc_net']:
                        feat_bonus = (feat_vals + 1) * 2.5
                    # For features in [0, 1] range, scale to [0, 5]
                    elif feat in ['vpin', 'smc_bullish', 'smc_bearish']:
                        feat_bonus = feat_vals * 5
                    # For velocity/trend features, use as-is capped
                    else:
                        feat_bonus = feat_vals.clip(-5, 5)
                    df['alpha_momentum_score'] += feat_bonus
                    df['phoenix_reversal_score'] += feat_bonus * 0.5  # Less for phoenix
                    boost_bonus += feat_bonus.mean()

            if boost_bonus > 0:
                print(f"  [v12] Feature boost applied: +{boost_bonus:.1f} avg points")

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

        # --- v10.6 CRITICAL FIX: Force validation tickers into top candidates ---
        # Without this, validation tickers with low ML scores (from zero features) never reach pattern analysis
        if ENABLE_VALIDATION_MODE:
            validation_tickers = (
                VALIDATION_SUITE['institutional_phoenix'] +
                VALIDATION_SUITE['speculative_phoenix']
            )
            forced_count = 0
            for ticker in validation_tickers:
                if ticker not in top_candidates['ticker'].values:
                    # Find ticker in full df
                    ticker_row = df[df['ticker'] == ticker]
                    if not ticker_row.empty:
                        top_candidates = pd.concat([top_candidates, ticker_row], ignore_index=True)
                        forced_count += 1
                        # Debug: Show LULU's actual ML score and rank
                        if ticker == 'LULU':
                            lulu_score = ticker_row.iloc[0].get('max_score', 0)
                            lulu_rank = (df['max_score'] > lulu_score).sum() + 1
                            print(f"  [VALIDATION] LULU ML score: {lulu_score:.2f}, rank: {lulu_rank}/{len(df)}")
                    else:
                        print(f"  [VALIDATION] ⚠️  {ticker} not in candidate pool (may be missing from data)")

            if forced_count > 0:
                print(f"  [VALIDATION] Force-added {forced_count} validation tickers to pattern analysis (bypassing top 75 filter)")
        # --- END v10.6 FIX ---

        print(f"  [INFO] Base scoring complete. Running pattern detection on {len(top_candidates)} candidates...")

        # --- PHASE 9: PATTERN DETECTION & EXPLANATION GENERATION ---
        pattern_results = {}
        tickers_to_analyze = top_candidates['ticker'].tolist()


        print(f"  [PATTERNS] Fetching 2-year price history for {len(tickers_to_analyze)} tickers via Alpaca...")

        # 1. Fetch data using our helper (fast, no sleep needed)
        start_date_pattern = datetime.now() - timedelta(days=730)
        price_data = fetch_alpaca_batch(tickers_to_analyze, start_date=start_date_pattern)

        # 2. Validation Debug Output
        if ENABLE_VALIDATION_MODE:
            validation_tickers = (
                VALIDATION_SUITE['institutional_phoenix'] +
                VALIDATION_SUITE['speculative_phoenix']
            )
            for vt in validation_tickers:
                if vt in tickers_to_analyze:
                    has_data = vt in price_data
                    data_len = len(price_data[vt]) if has_data else 0
                    print(f"  [VALIDATION] {vt} in analysis: data={'✓' if has_data else '✗'} ({data_len} rows)")


        # Run pattern detection on each candidate
        print(f"  [PATTERNS] Analyzing {len(tickers_to_analyze)} tickers for bull flags, GEX walls, and reversals...")

        # GEX Debug: Show overlap between candidates and strike gamma data
        gex_overlap = [t for t in tickers_to_analyze if t in self.strike_gamma_data]
        print(f"  [GEX DEBUG] {len(gex_overlap)}/{len(tickers_to_analyze)} candidates have strike gamma data")
        if gex_overlap and len(gex_overlap) <= 10:
            for t in gex_overlap[:5]:
                gamma_vals = list(self.strike_gamma_data[t].values())
                max_gamma = max(gamma_vals) if gamma_vals else 0
                print(f"    {t}: {len(gamma_vals)} strikes, max gamma {max_gamma/1000:.0f}K")
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
                'reversal': self.detect_downtrend_reversal(ticker, hist_df),
                'phoenix': self.detect_phoenix_reversal(ticker, hist_df),
                'cup_handle': self.detect_cup_and_handle(ticker, hist_df),
                'double_bottom': self.detect_double_bottom(ticker, hist_df)
            }
            pattern_results[ticker] = patterns

            # v10.6: Debug output for validation ticker phoenix results
            if ENABLE_VALIDATION_MODE:
                validation_tickers = VALIDATION_SUITE['institutional_phoenix'] + VALIDATION_SUITE['speculative_phoenix']
                if ticker in validation_tickers:
                    phoenix = patterns['phoenix']
                    phoenix_score = phoenix.get('phoenix_score', 0)
                    is_phoenix = phoenix.get('is_phoenix', False)
                    explanation = phoenix.get('explanation', 'N/A')[:80]
                    status = '✓ DETECTED' if is_phoenix else f'✗ Score={phoenix_score:.2f}/0.60'
                    print(f"  [VALIDATION] {ticker} Phoenix: {status}")
                    print(f"              → {explanation}")

            # --- PATTERN-BASED SCORE ADJUSTMENTS ---
            # v11.0: Now updates both legacy scores AND dual-ranking scores
            # v11.2 FIX: Accumulate phoenix_reversal_score properly (bonuses were overwriting each other!)

            # Initialize accumulators from base values
            phoenix_accum = row.get('phoenix_reversal_score', 0)
            alpha_accum = row.get('alpha_momentum_score', 0)
            trend_accum = row.get('trend_score_val', 0)
            ambush_accum = row.get('ambush_score_val', 0)

            # Bull flag bonus (MOMENTUM pattern - boosts alpha_momentum_score)
            flag_score = patterns['bull_flag'].get('flag_score', 0)
            if flag_score > 0:
                bonus = flag_score * 10  # Up to 10 point bonus
                trend_accum += bonus
                # v11.0: Bull flags are continuation patterns - boost alpha momentum
                alpha_accum += bonus * 1.5

            # GEX wall protection bonus (supports both strategies)
            wall_score = patterns['gex_wall'].get('wall_protection_score', 0)
            if wall_score > 0:
                bonus = wall_score * 8  # Up to 8 point bonus
                trend_accum += bonus
                ambush_accum += bonus
                alpha_accum += bonus
                phoenix_accum += bonus * 0.5

            # Reversal setup bonus (for ambush strategy)
            reversal_score = patterns['reversal'].get('reversal_score', 0)
            if reversal_score > 0:
                bonus = reversal_score * 12  # Up to 12 point bonus for ambush
                ambush_accum += bonus
                # v11.0: Reversals boost phoenix score
                phoenix_accum += bonus

            # Phoenix reversal bonus (CRITICAL for v11.0 - major boost to phoenix_reversal_score)
            phoenix_score = patterns['phoenix'].get('phoenix_score', 0)
            solidity_score = patterns['phoenix'].get('solidity_score', 0)
            if phoenix_score > 0:
                bonus = phoenix_score * 25  # v11.2: Increased from 15 to 25 for legacy scores
                trend_accum += bonus
                ambush_accum += bonus * 0.8
                # v11.0: MAJOR boost to phoenix_reversal_score - this is what fixes LULU ranking
                phoenix_boost = phoenix_score * 40  # v11.2: Increased from 25 to 40 for proper scaling
                phoenix_accum += phoenix_boost

            # v11.0: Solidity score bonus (institutional accumulation)
            if solidity_score > 0.3:
                solidity_bonus = solidity_score * 20  # v11.2: Increased from 15 to 20 for proper weight
                phoenix_accum += solidity_bonus
            top_candidates.at[idx, 'solidity_score'] = solidity_score

            # v11.2: Institutional Duration Bonus (12+ month bases)
            phoenix_data = patterns.get('phoenix', {})
            days_in_base = phoenix_data.get('days_in_base', 0)
            if days_in_base >= 365:  # Institutional threshold (12+ months)
                duration_bonus = min(20, (days_in_base / 365) * 10)  # Up to 20 pts for 730+ days
                phoenix_accum += duration_bonus

            # =========================================================================
            # v11.2: SOLIDITY GATE - Critical fix for false positives (e.g., MU at #1)
            # =========================================================================
            # True phoenix reversals REQUIRE clear institutional accumulation.
            # A stock with high dark pool activity but LOW solidity is likely a
            # momentum play, not a reversal from consolidation.
            #
            # Without this gate: MU with 0.40 solidity scored 99.9 (false positive)
            # With this gate: MU penalized, LULU with 0.70 solidity remains high
            # =========================================================================
            solidity_threshold = SOLIDITY_CONFIG.get('base_threshold', 0.55)
            if phoenix_score > 0 and solidity_score < solidity_threshold:
                # Apply 70% penalty - weak solidity = not a true phoenix reversal
                # This ensures momentum plays don't hijack the phoenix leaderboard
                penalty_factor = 0.30
                phoenix_accum = phoenix_accum * penalty_factor
                # Also reduce legacy scores to maintain consistency
                trend_accum = trend_accum * 0.7
                ambush_accum = ambush_accum * 0.7

            # =========================================================================
            # v11.5.2: MOMENTUM FILTER - Distinguish reversals from momentum plays
            # =========================================================================
            # True phoenix reversals emerge from extended consolidation BASES, not
            # from stocks already trading near their 52-week highs.
            #
            # CRITICAL FIX (v11.5.2): High solidity BYPASSES momentum penalty
            # - NVO (solidity 0.70) was incorrectly filtered despite valid accumulation
            # - High solidity = institutional conviction validates the breakout thesis
            # - Only penalize LOW solidity candidates near highs (e.g., MU at 0.40)
            #
            # Two-pronged check (with solidity gate):
            # 1. Near 52w high (within 20%) + LOW solidity = likely momentum, penalize
            # 2. Far from 52w low (>50% above) + LOW solidity = not a true reversal
            # =========================================================================
            pct_from_52w_high = phoenix_data.get('pct_from_52w_high', 0)
            pct_from_52w_low = phoenix_data.get('pct_from_52w_low', 0)
            solidity_threshold = 0.55  # Match SOLIDITY_CONFIG base_threshold

            # Check 1: Near 52-week high (only penalize LOW solidity candidates)
            # High solidity (>= 0.55) = institutional accumulation validates breakout
            if phoenix_score > 0 and pct_from_52w_high < 0.20 and solidity_score < solidity_threshold:
                # LOW solidity + near high = likely momentum play, not phoenix
                # Apply graduated penalty: closer to high = stronger penalty
                if pct_from_52w_high < 0.10:
                    momentum_penalty = 0.40  # Very close to high - strong penalty
                else:
                    momentum_penalty = 0.60  # 10-20% from high - moderate penalty
                phoenix_accum = phoenix_accum * momentum_penalty
                # Boost alpha score instead - this is a momentum play
                alpha_accum = alpha_accum * 1.2

            # Check 2: Too far from 52-week low (only penalize LOW solidity candidates)
            # High solidity + price recovery = working thesis, not false positive
            if phoenix_score > 0 and pct_from_52w_low > 0.50 and solidity_score < solidity_threshold:
                # LOW solidity + far from low = momentum dressed as phoenix
                # Apply penalty
                bottom_penalty = 0.70
                phoenix_accum = phoenix_accum * bottom_penalty

            # =========================================================================
            # v11.5.4: SECTOR BETA FILTER - Stock vs Sector Relative Performance
            # =========================================================================
            # True phoenix = stock OUTPERFORMING its sector (individual alpha)
            # Sector beta = stock just MATCHING sector (riding the wave)
            #
            # Example:
            # - LULU/NVO: Up 30% while sector up 10% → 20% outperformance → TRUE PHOENIX
            # - FCX/KGC: Up 25% while XLB up 25% → 0% outperformance → SECTOR BETA
            #
            # This is the correct filter because:
            # - It doesn't penalize all stocks when sectors are strong
            # - It specifically catches stocks just riding sector momentum
            # - True reversals should show INDIVIDUAL strength, not sector correlation
            # =========================================================================
            if phoenix_score > 0 and hasattr(self, 'sector_6m_returns') and self.sector_6m_returns:
                ticker_sector = self.sector_map_local.get(ticker, 'Unknown')
                sector_etf = SECTOR_MAP.get(ticker_sector)

                if sector_etf and sector_etf in self.sector_6m_returns:
                    sector_return = self.sector_6m_returns[sector_etf]

                    # Calculate stock's 6-month return from price history
                    # Use the cached price history from pattern analysis
                    stock_return = 0.0
                    if ticker in self.price_history_cache:
                        hist = self.price_history_cache[ticker]
                        if len(hist) >= 126:  # ~6 months of trading days
                            close_col = 'Close' if 'Close' in hist.columns else 'close'
                            if close_col in hist.columns:
                                start_price = hist[close_col].iloc[-126]
                                current_price = hist[close_col].iloc[-1]
                                stock_return = (current_price - start_price) / start_price if start_price > 0 else 0.0

                    # Calculate relative performance (alpha over sector)
                    relative_performance = stock_return - sector_return

                    # If stock is UNDERPERFORMING or BARELY MATCHING sector, penalize
                    # This catches FCX/KGC (matching XLB) but not LULU/NVO (outperforming)
                    if relative_performance < 0.05:  # Less than 5% outperformance
                        # Stock is just riding sector momentum - this is beta, not alpha
                        if relative_performance < -0.05:
                            # Stock underperforming sector - definitely not a leader
                            sector_beta_penalty = 0.40  # Heavy penalty
                        elif relative_performance < 0.0:
                            # Stock matching sector - likely beta play
                            sector_beta_penalty = 0.50
                        else:
                            # Stock barely outperforming (0-5%) - mild penalty
                            sector_beta_penalty = 0.70
                        phoenix_accum = phoenix_accum * sector_beta_penalty
                        # Boost alpha score - this belongs on momentum leaderboard
                        alpha_accum = alpha_accum * 1.2

            # =========================================================================
            # v11.5.5: REVERSAL RECENCY FILTER - Detect Early vs Late Stage Reversal
            # =========================================================================
            # Core insight: True phoenix = RECENT breakout from extended base
            # False positive = Prolonged uptrend (been rallying for months/years)
            #
            # Detection method: Compare 3-month vs 6-month returns
            # - "Flat then pop" pattern: 6mo ≈ 3mo returns (gains are RECENT) → TRUE PHOENIX
            # - "Up and up" pattern: 6mo >> 3mo returns (gains spread out) → PROLONGED TREND
            #
            # Example:
            # - LULU: 6mo=+40%, 3mo=+35% → 35/40=87.5% recency → early stage ✓
            # - FCX:  6mo=+50%, 3mo=+15% → 15/50=30% recency → prolonged trend ✗
            # =========================================================================
            if phoenix_score > 0 and hist_df is not None and len(hist_df) >= 126:

                # Handle both regular columns and MultiIndex columns (from yfinance)
                close_col = None
                if 'Close' in hist_df.columns:
                    close_col = 'Close'
                elif 'close' in hist_df.columns:
                    close_col = 'close'
                elif hasattr(hist_df.columns, 'get_level_values'):
                    # MultiIndex columns - try to find Close
                    level0 = hist_df.columns.get_level_values(0)
                    if 'Close' in level0:
                        close_col = 'Close'

                if close_col:  # Already checked len >= 126 above
                    # Get close prices, handling MultiIndex if needed
                    try:
                        if hasattr(hist_df.columns, 'nlevels') and hist_df.columns.nlevels > 1:
                            close_series = hist_df[close_col].iloc[:, 0] if isinstance(hist_df[close_col], pd.DataFrame) else hist_df[close_col]
                        else:
                            close_series = hist_df[close_col]

                        # Calculate 3-month return (last ~63 trading days)
                        return_3m = 0.0
                        if len(close_series) >= 63:
                            price_3m_ago = float(close_series.iloc[-63])
                            current_px = float(close_series.iloc[-1])
                            return_3m = (current_px - price_3m_ago) / price_3m_ago if price_3m_ago > 0 else 0

                        # Calculate 6-month return (last ~126 trading days)
                        return_6m = 0.0
                        if len(close_series) >= 126:
                            price_6m_ago = float(close_series.iloc[-126])
                            current_px = float(close_series.iloc[-1])
                            return_6m = (current_px - price_6m_ago) / price_6m_ago if price_6m_ago > 0 else 0

                        # Calculate "recency ratio" - what % of 6mo gains came in last 3mo?
                        # High ratio (>60%) = gains are RECENT = early stage breakout
                        # Low ratio (<40%) = gains spread over time = prolonged uptrend
                        if return_6m > 0.15:  # Only check stocks with meaningful 6mo gains
                            recency_ratio = return_3m / return_6m if return_6m > 0 else 0

                            # If recency ratio is LOW, gains are spread = prolonged uptrend = NOT early phoenix
                            if recency_ratio < 0.40:
                                # Gains spread over 6 months - this is a MATURE trend, not early reversal
                                # Strong penalty - this catches FCX/KGC (rallying for months/years)
                                recency_penalty = 0.35  # Heavy penalty for prolonged uptrends
                                phoenix_accum = phoenix_accum * recency_penalty
                                alpha_accum = alpha_accum * 1.3  # Boost momentum score instead
                            elif recency_ratio < 0.55:
                                # Moderate spread - partial penalty
                                recency_penalty = 0.60
                                phoenix_accum = phoenix_accum * recency_penalty
                                alpha_accum = alpha_accum * 1.15
                    except Exception:
                        pass  # Silently skip if price data is malformed

            # Cup-and-Handle bonus (hybrid pattern - continuation from base)
            cup_handle_score = patterns['cup_handle'].get('cup_handle_score', 0)
            if cup_handle_score > 0:
                bonus = cup_handle_score * 12  # Up to 12 point bonus
                trend_accum += bonus
                ambush_accum += bonus * 0.6
                # v11.0: Cup-handle is a reversal continuation pattern
                phoenix_accum += bonus

            # Double Bottom bonus (REVERSAL pattern - boosts phoenix_reversal_score)
            double_bottom_score = patterns['double_bottom'].get('double_bottom_score', 0)
            if double_bottom_score > 0:
                bonus = double_bottom_score * 10  # Up to 10 point bonus
                ambush_accum += bonus
                # v11.0: Double bottom is a reversal pattern - significant phoenix boost
                phoenix_accum += bonus * 1.5

            # --- PATTERN SYNERGY BONUSES (v10.4: LULU-inspired) ---
            # When multiple patterns overlap, it's a MUCH stronger signal
            # Phoenix + Double Bottom = Institutional accumulation with clear support
            if phoenix_score > 0 and double_bottom_score > 0:
                # LULU pattern: Extended base (phoenix) + double bottom support
                synergy_bonus = 8  # Significant bonus for dual pattern confirmation
                trend_accum += synergy_bonus
                ambush_accum += synergy_bonus
                # v11.0: This is THE LULU pattern - massive phoenix boost
                phoenix_accum += synergy_bonus * 2

            # Phoenix + Cup-Handle = Institutional accumulation with continuation setup
            elif phoenix_score > 0 and cup_handle_score > 0:
                synergy_bonus = 6
                trend_accum += synergy_bonus
                phoenix_accum += synergy_bonus * 1.5

            # Bull Flag + GEX Wall = Momentum with support
            elif flag_score > 0 and wall_score > 0.3:
                synergy_bonus = 5
                trend_accum += synergy_bonus
                alpha_accum += synergy_bonus * 1.5

            # v11.2 FIX: Write accumulated values back to DataFrame (single assignment, no overwrites)
            top_candidates.at[idx, 'phoenix_reversal_score'] = phoenix_accum
            top_candidates.at[idx, 'alpha_momentum_score'] = alpha_accum
            top_candidates.at[idx, 'trend_score_val'] = trend_accum
            top_candidates.at[idx, 'ambush_score_val'] = ambush_accum

            # Store pattern flags
            top_candidates.at[idx, 'has_bull_flag'] = patterns['bull_flag'].get('is_flag', False)
            top_candidates.at[idx, 'has_gex_support'] = wall_score > 0.3
            top_candidates.at[idx, 'is_reversal_setup'] = patterns['reversal'].get('is_reversal', False)
            top_candidates.at[idx, 'is_phoenix'] = patterns['phoenix'].get('is_phoenix', False)
            top_candidates.at[idx, 'is_cup_handle'] = patterns['cup_handle'].get('is_cup_handle', False)
            top_candidates.at[idx, 'is_double_bottom'] = patterns['double_bottom'].get('is_double_bottom', False)

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

        # v11.0: Format dual-ranking scores
        if 'alpha_momentum_score' in stock_candidates.columns:
            stock_candidates['alpha_score'] = stock_candidates['alpha_momentum_score'].clip(0, 99.9).round(1)
        else:
            stock_candidates['alpha_score'] = 0.0
        if 'phoenix_reversal_score' in stock_candidates.columns:
            stock_candidates['phoenix_score'] = stock_candidates['phoenix_reversal_score'].clip(0, 99.9).round(1)
        else:
            stock_candidates['phoenix_score'] = 0.0

        if 'dist_sma50' in stock_candidates.columns:
            stock_candidates['ext'] = (stock_candidates['dist_sma50'] * 100).round(1)
        else:
            stock_candidates['ext'] = 0.0

        # Ensure all expected columns exist
        for col in ['gamma_velocity', 'nn_score', 'stop_loss', 'take_profit', 'explanation', 'gex_support', 'solidity_score']:
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

        # --- v11.0: CREATE DUAL LEADERBOARDS ---
        print("\n  [v11.2] Creating Dual Leaderboards...")

        # Alpha Momentum Leaderboard (top 25 by alpha_momentum_score)
        alpha_config = DUAL_RANKING_CONFIG['alpha_momentum']
        alpha_leaderboard = stock_candidates.nlargest(alpha_config['top_n'], 'alpha_score')
        alpha_qualified = alpha_leaderboard[alpha_leaderboard['alpha_score'] >= alpha_config['min_score']]

        # Phoenix Reversal Leaderboard (top 25 by phoenix_reversal_score)
        phoenix_config = DUAL_RANKING_CONFIG['phoenix_reversal']
        phoenix_leaderboard = stock_candidates.nlargest(phoenix_config['top_n'], 'phoenix_score')
        phoenix_qualified = phoenix_leaderboard[phoenix_leaderboard['phoenix_score'] >= phoenix_config['min_score']]

        # Store leaderboards for output
        stock_candidates.attrs['alpha_leaderboard'] = alpha_qualified
        stock_candidates.attrs['phoenix_leaderboard'] = phoenix_qualified

        print(f"  [v11.2] Alpha Momentum: {len(alpha_qualified)} qualified (>= {alpha_config['min_score']})")
        print(f"  [v11.2] Phoenix Reversal: {len(phoenix_qualified)} qualified (>= {phoenix_config['min_score']})")

        # --- SUMMARY STATISTICS ---
        bull_flags = stock_candidates['has_bull_flag'].sum() if 'has_bull_flag' in stock_candidates.columns else 0
        gex_protected = stock_candidates['has_gex_support'].sum() if 'has_gex_support' in stock_candidates.columns else 0
        reversal_setups = stock_candidates['is_reversal_setup'].sum() if 'is_reversal_setup' in stock_candidates.columns else 0
        phoenix_reversals = stock_candidates['is_phoenix'].sum() if 'is_phoenix' in stock_candidates.columns else 0
        cup_handles = stock_candidates['is_cup_handle'].sum() if 'is_cup_handle' in stock_candidates.columns else 0
        double_bottoms = stock_candidates['is_double_bottom'].sum() if 'is_double_bottom' in stock_candidates.columns else 0

        print(f"\n  [PATTERNS SUMMARY]")
        print(f"    Bull Flags Detected: {bull_flags}")
        print(f"    GEX Protected Positions: {gex_protected}")
        print(f"    Reversal Setups: {reversal_setups}")
        print(f"    Phoenix Reversals: {phoenix_reversals}")
        print(f"    Cup-and-Handle Patterns: {cup_handles}")
        print(f"    Double Bottom Patterns: {double_bottoms}")

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

    # FIX: Pass Google Drive path as base_dir for database persistence
    # Without this, DB defaults to /content/ which is ephemeral in Colab
    data_dir = "/content/drive/My Drive/colab"
    if not os.path.exists(data_dir): data_dir = os.getcwd()

    engine = SwingTradingEngine(base_dir=data_dir)
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

            # --- POSITION SIZING (Phase 10: Kelly Criterion) ---
            print(f"\n  [SIZING] Calculating Kelly-based position sizes...")
            portfolio_value = 100000  # Default portfolio size
            for idx, row in stocks_df.iterrows():
                sizing = engine.calculate_position_size(row.to_dict(), portfolio_value)
                stocks_df.at[idx, 'position_pct'] = sizing['position_pct']
                stocks_df.at[idx, 'position_size'] = sizing['position_size']
                stocks_df.at[idx, 'shares'] = sizing['shares']
                stocks_df.at[idx, 'risk_tier'] = sizing['risk_tier']

            # NOTE: save_db() is called at the end of the main block, not here
            # Calling it twice would close the connection prematurely

            # --- GENERATE FULL REPORT (See engine/report.py for ~250 lines) ---
            elapsed = time.time() - start_time
            msg, alpha_picks, phoenix_picks = _generate_full_report(
                stocks_df, etfs_df, engine, DUAL_RANKING_CONFIG,
                ENABLE_VALIDATION_MODE, elapsed, device_name
            )

            # --- SAVE OUTPUT ---
            out_path = "/content/drive/My Drive/colab/swing_signals_v11_grandmaster.csv" if COLAB_ENV else os.path.join(engine.base_dir, "swing_signals_v11_grandmaster.csv")
            try:
                final_output = pd.concat([stocks_df, etfs_df])
                final_output.to_csv(out_path, index=False)
                print(f"\n[SUCCESS] Saved comprehensive report with dual leaderboards: {out_path}")
            except Exception as e:
                out_path = os.path.join(engine.base_dir, "swing_signals_v11_grandmaster.csv")
                try:
                    pd.concat([stocks_df, etfs_df]).to_csv(out_path, index=False)
                    print(f"\n[FALLBACK] Saved locally to {out_path}")
                except:
                    print(f"\n[ERROR] Could not save output: {e}")

            # Log run history
            try:
                hist_path = "/content/drive/My Drive/colab/run_history.txt" if COLAB_ENV else os.path.join(engine.base_dir, "run_history.txt")
                with open(hist_path, "a") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
            except:
                pass

            # Persist DB to Google Drive
            engine.history_mgr.save_db()
    else:
        print("[CRITICAL] Missing data files. Please ensure Unusual Whales data is in the expected location.")
        # Still save DB even if data files missing (preserves any synced history)
        engine.history_mgr.save_db()
