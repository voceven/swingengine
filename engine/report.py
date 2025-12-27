# -*- coding: utf-8 -*-
"""
engine/report.py - Report Generation Functions for Swing Trading Engine v12

Extracted report printing functions to reduce main file size.
Each function handles printing a specific section of the output report.
"""

import pandas as pd
from datetime import datetime


def print_header(market_regime, macro_data=None):
    """Print the report header with macro context."""
    print("\n" + "="*80)
    print(f"GRANDMASTER ENGINE v11.2 - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*80)
    print(f"MACRO REGIME: {market_regime}")
    if macro_data:
        print(f"  VIX: {macro_data.get('vix', 0):.2f} | TNX: {macro_data.get('tnx', 0):.2f}% | DXY: {macro_data.get('dxy', 0):.2f}")
        print(f"  Score Adjustment: {macro_data.get('adjustment', 0):+.1f} points")


def print_alpha_leaderboard(stocks_df, dual_ranking_config, enable_validation=False):
    """Print Alpha Momentum leaderboard."""
    print("\n" + "="*80)
    print("🚀 LEADERBOARD 1: ALPHA MOMENTUM (Continuation Plays)")
    print("Logic: Trend + ML Ensemble + Neural Network + Volume Momentum")
    print("Use for: Riding existing trends, momentum breakouts, bull flags")
    print("="*80)

    if 'alpha_score' in stocks_df.columns:
        alpha_min = dual_ranking_config['alpha_momentum']['min_score']
        alpha_picks = stocks_df[stocks_df['alpha_score'] >= alpha_min].nlargest(25, 'alpha_score')
    else:
        alpha_picks = pd.DataFrame()

    display_cols = ['ticker', 'alpha_score', 'quality', 'has_bull_flag', 'nn_score', 'position_pct', 'risk_tier']
    display_cols = [c for c in display_cols if c in stocks_df.columns]

    if not alpha_picks.empty:
        print(alpha_picks[display_cols].head(10).to_string(index=False))
        print(f"\n  Total Alpha Momentum candidates: {len(alpha_picks)}")
    else:
        print("  [!] No candidates met Alpha Momentum criteria (>= 75)")

    return alpha_picks


def print_phoenix_leaderboard(stocks_df, dual_ranking_config, enable_validation=False):
    """Print Phoenix Reversals leaderboard."""
    print("\n" + "="*80)
    print("🔥 LEADERBOARD 2: PHOENIX REVERSALS (Base Breakout Plays)")
    print("Logic: Solidity Score + Consolidation Duration + Institutional Flow")
    print("Use for: LULU-style setups, deep value reversals, accumulation breakouts")
    print("="*80)

    if 'phoenix_score' in stocks_df.columns:
        phoenix_min = dual_ranking_config['phoenix_reversal']['min_score']
        phoenix_picks = stocks_df[stocks_df['phoenix_score'] >= phoenix_min].nlargest(25, 'phoenix_score')
    else:
        phoenix_picks = pd.DataFrame()

    display_cols = ['ticker', 'phoenix_score', 'solidity_score', 'is_phoenix', 'quality', 'dp_support', 'position_pct']
    display_cols = [c for c in display_cols if c in stocks_df.columns]

    if not phoenix_picks.empty:
        print(phoenix_picks[display_cols].head(10).to_string(index=False))

        print("\n  --- PHOENIX SIGNAL EXPLANATIONS ---")
        for _, row in phoenix_picks.head(5).iterrows():
            ticker = row.get('ticker', 'N/A')
            phoenix_s = row.get('phoenix_score', 0)
            solidity_s = row.get('solidity_score', 0)
            explanation = str(row.get('explanation', 'N/A'))[:97] + "..." if len(str(row.get('explanation', ''))) > 100 else row.get('explanation', 'N/A')
            print(f"  {ticker} [P:{phoenix_s:.1f} S:{solidity_s:.2f}]: {explanation}")

        print(f"\n  Total Phoenix Reversal candidates: {len(phoenix_picks)}")

        # LULU validation
        if enable_validation and 'LULU' in phoenix_picks['ticker'].values:
            lulu_row = phoenix_picks[phoenix_picks['ticker'] == 'LULU'].iloc[0]
            lulu_rank = phoenix_picks.index.get_loc(phoenix_picks[phoenix_picks['ticker'] == 'LULU'].index[0]) + 1
            print(f"\n  [v11.0 VALIDATION] LULU Phoenix Rank: #{lulu_rank}/25")
            print(f"                    Phoenix Score: {lulu_row.get('phoenix_score', 0):.1f}")
            print(f"                    Solidity Score: {lulu_row.get('solidity_score', 0):.2f}")
    else:
        print("  [!] No candidates met Phoenix Reversal criteria (>= 60)")

    return phoenix_picks


def print_momentum_strategy(stocks_df):
    """Print Strategy 1: Momentum Leaders."""
    trend_picks = stocks_df[stocks_df['trend_score'] > 80].sort_values('trend_score', ascending=False)

    print("\n" + "="*80)
    print(f"[LEGACY] STRATEGY 1: MOMENTUM LEADERS (Trend Following)")
    print(f"Logic: CatBoost + Hive Mind + Bull Flag Detection + GEX Protection")
    print("="*80)

    display_cols = ['ticker', 'trend_score', 'quality', 'has_bull_flag', 'nn_score', 'position_pct', 'shares', 'risk_tier']
    display_cols = [c for c in display_cols if c in stocks_df.columns]

    if not trend_picks.empty:
        print(trend_picks[display_cols].head(8).to_string(index=False))

        print("\n  --- POSITION SIZING (Kelly) ---")
        for _, row in trend_picks.head(5).iterrows():
            pct = row.get('position_pct', 0)
            shares = int(row.get('shares', 0))
            tier = row.get('risk_tier', 'N/A')
            price = row.get('current_price', 0)
            print(f"  {row['ticker']}: {pct:.1f}% (${pct*1000:.0f} → {shares} shares @ ${price:.2f}) | {tier}")

        print("\n  --- SIGNAL EXPLANATIONS ---")
        for _, row in trend_picks.head(5).iterrows():
            explanation = str(row.get('explanation', 'N/A'))[:117] + "..." if len(str(row.get('explanation', ''))) > 120 else row.get('explanation', 'N/A')
            print(f"  {row['ticker']}: {explanation}")
    else:
        print("  [!] No candidates met strict criteria (>80).")

    return trend_picks


def print_ambush_strategy(stocks_df):
    """Print Strategy 2: Ambush Predators."""
    ambush_picks = stocks_df[
        (stocks_df['ambush_score'] > 75) &
        ((stocks_df['is_reversal_setup'] == True) | (stocks_df['sector_status'] == 'Lagging Sector'))
    ].sort_values('ambush_score', ascending=False)

    print("\n" + "="*80)
    print(f"[LEGACY] STRATEGY 2: AMBUSH PREDATORS (Counter-Trend Reversals)")
    print(f"Logic: Downtrend + DP Support + Divergence Detection")
    print("="*80)

    display_cols = ['ticker', 'ambush_score', 'quality', 'is_reversal_setup', 'dp_support', 'gex_support', 'position_pct', 'shares']
    display_cols = [c for c in display_cols if c in stocks_df.columns]

    if not ambush_picks.empty:
        print(ambush_picks[display_cols].head(8).to_string(index=False))

        print("\n  --- SIGNAL EXPLANATIONS ---")
        for _, row in ambush_picks.head(5).iterrows():
            explanation = str(row.get('explanation', 'N/A'))[:117] + "..." if len(str(row.get('explanation', ''))) > 120 else row.get('explanation', 'N/A')
            print(f"  {row['ticker']}: {explanation}")
    else:
        print("  [!] No reversal candidates found.")

    return ambush_picks


def print_bull_flag_strategy(stocks_df):
    """Print Strategy 3: Bull Flag Breakouts."""
    flag_picks = stocks_df[stocks_df['has_bull_flag'] == True].sort_values('trend_score', ascending=False)

    if not flag_picks.empty:
        print("\n" + "="*80)
        print(f"STRATEGY 3: BULL FLAG BREAKOUTS")
        print(f"Logic: Strong pole + Tight consolidation + Declining volume")
        print("="*80)

        display_cols = ['ticker', 'trend_score', 'quality', 'nn_score', 'position_pct', 'shares', 'stop_loss', 'take_profit']
        display_cols = [c for c in display_cols if c in stocks_df.columns]
        print(flag_picks[display_cols].head(5).to_string(index=False))

        print("\n  --- SIGNAL EXPLANATIONS ---")
        for _, row in flag_picks.head(3).iterrows():
            explanation = str(row.get('explanation', 'N/A'))[:117] + "..." if len(str(row.get('explanation', ''))) > 120 else row.get('explanation', 'N/A')
            print(f"  {row['ticker']}: {explanation}")

    return flag_picks


def print_etf_strategy(etfs_df):
    """Print Strategy 4: ETF Swing."""
    if not etfs_df.empty:
        print("\n" + "="*80)
        print(f"STRATEGY 4: ETF SWING")
        print("="*80)

        etf_display = ['ticker', 'trend_score_val', 'sector_status', 'net_gamma', 'nn_score', 'stop_loss', 'take_profit']
        etf_display = [c for c in etf_display if c in etfs_df.columns]
        print(etfs_df[etf_display].head(5).to_string(index=False))


def print_phoenix_strategy(stocks_df):
    """Print Strategy 5: Phoenix Reversal."""
    phoenix_df = stocks_df[stocks_df['is_phoenix'] == True].copy() if 'is_phoenix' in stocks_df.columns else pd.DataFrame()

    print("\n" + "="*80)
    print(f"STRATEGY 5: PHOENIX REVERSAL (6-12 Month Base Breakouts)")
    print("Logic: Extended consolidation + Volume surge + RSI 50-70 + Breakout")
    print("="*80)

    if not phoenix_df.empty:
        phoenix_display = ['ticker', 'phoenix_score', 'solidity_score', 'quality', 'nn_score', 'position_pct', 'shares', 'stop_loss', 'take_profit']
        phoenix_display = [c for c in phoenix_display if c in phoenix_df.columns]
        phoenix_sorted = phoenix_df.sort_values('phoenix_score', ascending=False).head(5)
        print(phoenix_sorted[phoenix_display].to_string(index=False))

        print("\n  --- SIGNAL EXPLANATIONS ---")
        for _, row in phoenix_sorted.iterrows():
            expl = str(row.get('explanation', 'No explanation'))[:150] + "..."
            print(f"  {row['ticker']}: {expl}")
    else:
        print("  [!] No phoenix reversal candidates found.")

    return phoenix_df


def print_summary(stocks_df, etfs_df, alpha_picks, phoenix_picks, trend_picks, ambush_picks,
                  flag_picks, phoenix_df, elapsed_time, device_name):
    """Print run summary."""
    mins, secs = divmod(elapsed_time, 60)

    print("\n" + "="*80)
    print("RUN SUMMARY - v11.2 DUAL-RANKING ARCHITECTURE")
    print("="*80)
    print(f"  Duration: {int(mins)}m {int(secs)}s")
    print(f"  Device: {device_name}")
    print(f"  Total Candidates Analyzed: {len(stocks_df) + len(etfs_df)}")

    alpha_count = len(alpha_picks) if not alpha_picks.empty else 0
    phoenix_lb_count = len(phoenix_picks) if not phoenix_picks.empty else 0

    print(f"\n  --- v11.0 DUAL LEADERBOARDS ---")
    print(f"  Alpha Momentum Leaders: {alpha_count}")
    print(f"  Phoenix Reversals: {phoenix_lb_count}")

    print(f"\n  --- Legacy Strategies ---")
    print(f"  Momentum Leaders (>80): {len(trend_picks)}")
    print(f"  Ambush Setups: {len(ambush_picks)}")
    print(f"  Bull Flags: {len(flag_picks) if not flag_picks.empty else 0}")
    print(f"  Phoenix Patterns: {len(phoenix_df) if not phoenix_df.empty else 0}")

    msg = f"v11.2 | Duration: {int(mins)}m {int(secs)}s | Device: {device_name} | Alpha: {alpha_count} | Phoenix: {phoenix_lb_count}"
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- {msg} ---")

    return msg


def generate_full_report(stocks_df, etfs_df, engine, dual_ranking_config,
                         enable_validation, elapsed_time, device_name):
    """Generate the complete report with all sections."""
    # Header
    macro_data = getattr(engine, 'macro_data', None)
    print_header(engine.market_regime, macro_data)

    # Dual leaderboards
    alpha_picks = print_alpha_leaderboard(stocks_df, dual_ranking_config, enable_validation)
    phoenix_picks = print_phoenix_leaderboard(stocks_df, dual_ranking_config, enable_validation)

    # Legacy strategies
    trend_picks = print_momentum_strategy(stocks_df)
    ambush_picks = print_ambush_strategy(stocks_df)
    flag_picks = print_bull_flag_strategy(stocks_df)
    print_etf_strategy(etfs_df)
    phoenix_df = print_phoenix_strategy(stocks_df)

    # Summary
    msg = print_summary(stocks_df, etfs_df, alpha_picks, phoenix_picks, trend_picks,
                        ambush_picks, flag_picks, phoenix_df, elapsed_time, device_name)

    return msg, alpha_picks, phoenix_picks
