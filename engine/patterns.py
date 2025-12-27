"""
engine/patterns.py - Pattern Detection Functions for Swing Trading Engine v12

Extracted pattern detection functions that take required data as parameters
instead of relying on instance state. This enables modular testing and
reduces the main engine file size.

Functions:
- detect_bull_flag: Bull flag pattern detection
- find_gex_walls: GEX wall support/resistance detection
- detect_downtrend_reversal: Downtrend reversal setup detection
- calculate_flow_factor: Institutional flow factor calculation
- calculate_solidity_score: Institutional accumulation score
- detect_phoenix_reversal: Phoenix reversal pattern detection
- detect_cup_and_handle: Cup-and-handle pattern detection
- detect_double_bottom: Double bottom pattern detection
"""

import numpy as np
import pandas as pd
import math

# Import config values
from .config import (
    BULL_FLAG_CONFIG, GEX_WALL_CONFIG, REVERSAL_CONFIG,
    PHOENIX_CONFIG, SOLIDITY_CONFIG, GATEKEEPER_CONFIG,
    ENABLE_VALIDATION_MODE, VALIDATION_SUITE
)


def detect_bull_flag(history_df, config=None):
    """
    Detect bull flag pattern: strong upward move (pole) followed by consolidation (flag).

    Args:
        history_df: DataFrame with OHLCV data
        config: Optional config override (defaults to BULL_FLAG_CONFIG)

    Returns:
        dict with is_flag, flag_score, pole_gain, flag_range, explanation
    """
    if config is None:
        config = BULL_FLAG_CONFIG

    result = {
        'is_flag': False,
        'flag_score': 0.0,
        'pole_gain': 0.0,
        'flag_range': 0.0,
        'explanation': ''
    }

    lookback = config['pole_days'] + config['flag_days'] + 5

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

        close = close.iloc[-lookback:]
        if volume is not None:
            volume = volume.iloc[-lookback:]

        pole_days = config['pole_days']
        flag_days = config['flag_days']

        # Pole phase
        pole_start = close.iloc[0]
        pole_end = close.iloc[pole_days]
        pole_gain = (pole_end - pole_start) / pole_start if pole_start > 0 else 0

        # Flag phase
        flag_segment = close.iloc[-flag_days:]
        flag_high = flag_segment.max()
        flag_low = flag_segment.min()
        flag_range = (flag_high - flag_low) / flag_segment.mean() if flag_segment.mean() > 0 else 1

        # Volume analysis
        volume_declining = True
        volume_ratio = 1.0
        if volume is not None and len(volume) >= lookback:
            pole_volume = volume.iloc[:pole_days].mean()
            flag_volume = volume.iloc[-flag_days:].mean()
            if pole_volume > 0:
                volume_ratio = flag_volume / pole_volume
                volume_declining = volume_ratio < config['volume_decline_ratio']

        # Check pattern
        pole_ok = pole_gain >= config['pole_min_gain']
        flag_ok = flag_range <= config['flag_max_range']
        is_flag = pole_ok and flag_ok

        # v11.5 FIX: Check if breakout already occurred
        current_price = close.iloc[-1]
        breakout_threshold = 1.05
        breakout_already_happened = current_price > flag_high * breakout_threshold

        if breakout_already_happened and is_flag:
            is_flag = False
            breakout_pct = ((current_price / flag_high) - 1) * 100
            result['is_flag'] = False
            result['pole_gain'] = pole_gain
            result['flag_range'] = flag_range
            result['flag_score'] = 0.1
            result['explanation'] = f"Flag breakout already occurred (+{breakout_pct:.1f}% above flag)"
            return result

        result['pole_gain'] = pole_gain
        result['flag_range'] = flag_range
        result['is_flag'] = is_flag

        if is_flag:
            base_score = min(1.0, (pole_gain / 0.15) * 0.4 + (1 - flag_range / config['flag_max_range']) * 0.4)
            if volume_declining:
                base_score += 0.2
            result['flag_score'] = base_score
            result['explanation'] = f"BULL FLAG: {pole_gain*100:.1f}% pole, {flag_range*100:.1f}% range"
            if volume_declining:
                result['explanation'] += " +vol"
        elif pole_ok and not flag_ok:
            result['flag_score'] = 0.25
            result['explanation'] = f"Flag forming: {pole_gain*100:.1f}% pole, range {flag_range*100:.1f}% (need <{config['flag_max_range']*100:.0f}%)"
        elif pole_gain > 0.04:
            result['flag_score'] = 0.15
            result['explanation'] = f"Weak momentum: {pole_gain*100:.1f}% move"
        else:
            result['explanation'] = 'No bull flag pattern'

    except Exception as e:
        result['explanation'] = f'Pattern detection error: {str(e)[:50]}'

    return result


def find_gex_walls(ticker, current_price, strike_gamma_data=None, dp_support_levels=None, bot_df=None):
    """
    Find gamma exposure (GEX) walls that act as support/resistance.

    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        strike_gamma_data: Dict mapping tickers to {strike: gamma} dicts
        dp_support_levels: Dict mapping tickers to list of DP support prices
        bot_df: Optional DataFrame with strike-level gamma data

    Returns:
        dict with support_wall, resistance_wall, wall_protection_score, explanation
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
    if strike_gamma_data and ticker in strike_gamma_data and strike_gamma_data[ticker]:
        try:
            strike_gamma = strike_gamma_data[ticker]

            # Support walls
            support_candidates = {k: v for k, v in strike_gamma.items()
                                 if k < current_price and v > GEX_WALL_CONFIG['min_support_gamma']}

            if support_candidates:
                best_strike = max(support_candidates.keys(), key=lambda x: support_candidates[x])
                result['support_wall'] = float(best_strike)
                support_gamma = support_candidates[best_strike]
                proximity = (current_price - best_strike) / current_price

                if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                    result['wall_protection_score'] = min(1.0, support_gamma / 250_000)

            # Resistance walls
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

    # PRIORITY 2: Check strike-level data in bot_df
    if bot_df is not None and not bot_df.empty and 'strike' in bot_df.columns:
        try:
            ticker_data = bot_df[bot_df['ticker'] == ticker] if 'ticker' in bot_df.columns else bot_df

            if ticker_data.empty or 'net_gamma' not in ticker_data.columns:
                result['explanation'] = 'No gamma data for strike analysis'
                return result

            strike_gamma = ticker_data.groupby('strike')['net_gamma'].sum()

            support_mask = (strike_gamma.index < current_price) & (strike_gamma > GEX_WALL_CONFIG['min_support_gamma'])
            support_candidates = strike_gamma[support_mask]

            if not support_candidates.empty:
                result['support_wall'] = float(support_candidates.idxmax())
                support_gamma = support_candidates.max()
                proximity = (current_price - result['support_wall']) / current_price

                if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                    result['wall_protection_score'] = min(1.0, support_gamma / 250_000)

            resist_mask = (strike_gamma.index > current_price) & (strike_gamma < GEX_WALL_CONFIG['min_resist_gamma'])
            resist_candidates = strike_gamma[resist_mask]

            if not resist_candidates.empty:
                result['resistance_wall'] = float(resist_candidates.idxmin())

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
    if dp_support_levels and ticker in dp_support_levels:
        dp_levels = dp_support_levels[ticker]
        support_levels = [p for p in dp_levels if p < current_price]
        if support_levels:
            best_support = max(support_levels)
            proximity = (current_price - best_support) / current_price
            if proximity <= GEX_WALL_CONFIG['proximity_pct']:
                result['support_wall'] = best_support
                result['wall_protection_score'] = 0.5
                result['explanation'] = f"DP support at ${best_support:.2f} ({proximity*100:.1f}% below)"
            else:
                result['explanation'] = f"DP support at ${best_support:.2f} (too far: {proximity*100:.1f}%)"
        else:
            result['explanation'] = 'No support levels detected'
    else:
        result['explanation'] = 'No strike-level or DP data available'

    return result


def detect_downtrend_reversal(ticker, history_df, dp_support_levels=None):
    """
    Detect potential reversal setup: months-long downtrend with dark pool support.

    Args:
        ticker: Stock ticker symbol
        history_df: DataFrame with OHLCV data
        dp_support_levels: Dict mapping tickers to list of DP support prices

    Returns:
        dict with is_reversal, reversal_score, days_below_sma, has_dp_support, explanation
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

        sma50 = close.rolling(50).mean()

        recent_close = close.iloc[-lookback:]
        recent_sma = sma50.iloc[-lookback:]
        days_below = (recent_close < recent_sma).sum()
        result['days_below_sma'] = int(days_below)

        is_downtrend = days_below >= REVERSAL_CONFIG['min_days_below_sma']

        current_price = float(close.iloc[-1])
        has_dp_support = False
        dp_level = None

        if dp_support_levels and ticker in dp_support_levels:
            dp_levels = dp_support_levels[ticker]
            nearby_support = [p for p in dp_levels if abs(p - current_price) / current_price < REVERSAL_CONFIG['dp_proximity_pct']]
            if nearby_support:
                has_dp_support = True
                dp_level = max(nearby_support)

        result['has_dp_support'] = has_dp_support

        # RSI divergence check
        has_divergence = False
        if len(close) > 20:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

            price_trend = close.iloc[-10:].mean() < close.iloc[-20:-10].mean()
            rsi_trend = rsi.iloc[-10:].mean() > rsi.iloc[-20:-10].mean()
            has_divergence = price_trend and rsi_trend

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


def calculate_flow_factor(ticker, volume_ratio, full_df=None, signature_prints=None):
    """
    Calculate Flow Factor (0.0 to 1.0) for dynamic threshold adjustment.

    Args:
        ticker: Stock ticker symbol
        volume_ratio: Recent volume / average volume ratio
        full_df: DataFrame with ticker data (dp_total, net_gamma, net_delta)
        signature_prints: Dict mapping tickers to signature print data

    Returns:
        tuple: (flow_factor, flow_details dict)
    """
    flow_factor = 0.0
    flow_details = {
        'volume_component': 0.0,
        'dp_component': 0.0,
        'signature_component': 0.0,
        'options_component': 0.0,
        'raw_dp_total': 0.0,
        'has_signature': False
    }

    try:
        # Component 1: Volume Intensity (0-0.25)
        if volume_ratio >= 1.0:
            if volume_ratio >= 4.0:
                flow_details['volume_component'] = 0.25
            elif volume_ratio >= 2.5:
                flow_details['volume_component'] = 0.20
            elif volume_ratio >= 1.5:
                flow_details['volume_component'] = 0.15
            else:
                flow_details['volume_component'] = 0.05 + (volume_ratio - 1.0) * 0.10

        # Component 2: Dark Pool Activity (0-0.35)
        ticker_data = None
        if full_df is not None and not full_df.empty:
            ticker_rows = full_df[full_df['ticker'] == ticker]
            if not ticker_rows.empty:
                ticker_data = ticker_rows.iloc[0]

        if ticker_data is not None and 'dp_total' in ticker_data:
            dp_total = float(ticker_data.get('dp_total', 0))
            flow_details['raw_dp_total'] = dp_total

            if dp_total > 0:
                if dp_total >= 1_000_000_000:
                    flow_details['dp_component'] = 0.35
                elif dp_total >= 500_000_000:
                    flow_details['dp_component'] = 0.30
                elif dp_total >= 100_000_000:
                    flow_details['dp_component'] = 0.25
                elif dp_total >= 50_000_000:
                    flow_details['dp_component'] = 0.20
                elif dp_total >= 10_000_000:
                    flow_details['dp_component'] = 0.12
                elif dp_total >= 1_000_000:
                    flow_details['dp_component'] = 0.05

        # Component 3: Institutional Signature Prints (0-0.20)
        if signature_prints and ticker in signature_prints:
            flow_details['has_signature'] = True
            sig_count = len(signature_prints[ticker]) if isinstance(signature_prints[ticker], list) else 1
            flow_details['signature_component'] = min(0.20, 0.10 + sig_count * 0.05)
        elif ticker_data is not None:
            dp_max = float(ticker_data.get('dp_max', ticker_data.get('dp_total', 0)))
            if dp_max >= 50_000_000:
                flow_details['signature_component'] = 0.15
                flow_details['has_signature'] = True
            elif dp_max >= 10_000_000:
                flow_details['signature_component'] = 0.08

        # Component 4: Net Options Flow (0-0.20)
        if ticker_data is not None:
            net_gamma = float(ticker_data.get('net_gamma', 0))
            net_delta = float(ticker_data.get('net_delta', 0))

            if net_gamma > 0 and net_delta > 0:
                gamma_strength = min(1.0, abs(net_gamma) / 1000000)
                delta_strength = min(1.0, abs(net_delta) / 5000000)
                flow_details['options_component'] = 0.20 * (gamma_strength + delta_strength) / 2
            elif net_gamma > 0 or net_delta > 0:
                flow_details['options_component'] = 0.08

        flow_factor = min(1.0,
            flow_details['volume_component'] +
            flow_details['dp_component'] +
            flow_details['signature_component'] +
            flow_details['options_component']
        )

    except Exception:
        flow_factor = 0.0

    return flow_factor, flow_details


def calculate_solidity_score(ticker, history_df, full_df=None, dp_support_levels=None):
    """
    Calculate Solidity Score - detects institutional accumulation during retail exhaustion.

    Args:
        ticker: Stock ticker symbol
        history_df: DataFrame with OHLCV data
        full_df: DataFrame with ticker data (dp_total)
        dp_support_levels: Dict mapping tickers to list of DP support prices

    Returns:
        dict with solidity_score, components, is_solid, explanation
    """
    result = {
        'solidity_score': 0.0,
        'consolidation_quality': 0.0,
        'volume_decline': 0.0,
        'institutional_flow': 0.0,
        'duration_score': 0.0,
        'is_solid': False,
        'explanation': ''
    }

    if history_df is None or len(history_df) < SOLIDITY_CONFIG['min_consolidation_days']:
        result['explanation'] = 'Insufficient history for solidity analysis'
        return result

    try:
        close = history_df['Close']
        volume = history_df['Volume']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        # Component 1: Consolidation Quality (0-0.30)
        lookback = min(len(close), 60)
        recent_close = close.iloc[-lookback:]

        base_high = recent_close.max()
        base_low = recent_close.min()
        price_range = (base_high - base_low) / base_low if base_low > 0 else 1.0

        fib_threshold = SOLIDITY_CONFIG['max_consolidation_range']

        if price_range <= fib_threshold * 0.5:
            result['consolidation_quality'] = 0.30
        elif price_range <= fib_threshold:
            result['consolidation_quality'] = 0.25
        elif price_range <= fib_threshold * 1.5:
            result['consolidation_quality'] = 0.15
        else:
            result['consolidation_quality'] = 0.0

        # Component 2: Volume Decline (0-0.25)
        vol_lookback = SOLIDITY_CONFIG['volume_lookback_days']
        if len(volume) >= vol_lookback:
            avg_volume_long = volume.iloc[-vol_lookback:].mean()
            avg_volume_recent = volume.iloc[-5:].mean()

            if avg_volume_long > 0:
                volume_ratio = avg_volume_recent / avg_volume_long

                if volume_ratio <= SOLIDITY_CONFIG['volume_decline_ratio'] * 0.5:
                    result['volume_decline'] = 0.25
                elif volume_ratio <= SOLIDITY_CONFIG['volume_decline_ratio']:
                    result['volume_decline'] = 0.20
                elif volume_ratio <= 1.0:
                    result['volume_decline'] = 0.10
                else:
                    result['volume_decline'] = 0.0

        # Component 3: Institutional Flow (0-0.25)
        dp_total = 0
        has_signature = False

        if full_df is not None and not full_df.empty:
            ticker_data = full_df[full_df['ticker'] == ticker]
            if not ticker_data.empty:
                dp_total = ticker_data.iloc[0].get('dp_total', 0)

        if dp_support_levels and ticker in dp_support_levels:
            has_signature = len(dp_support_levels[ticker]) > 0

        min_dp = SOLIDITY_CONFIG['min_dp_total']
        if dp_total >= min_dp * 5:
            result['institutional_flow'] = 0.25
        elif dp_total >= min_dp * 2:
            result['institutional_flow'] = 0.20
        elif dp_total >= min_dp:
            result['institutional_flow'] = 0.15
        elif has_signature:
            result['institutional_flow'] = 0.10
        else:
            result['institutional_flow'] = 0.0

        if has_signature and result['institutional_flow'] > 0:
            result['institutional_flow'] = min(0.25,
                result['institutional_flow'] + SOLIDITY_CONFIG['signature_print_bonus'])

        # Component 4: Duration Score (0-0.20)
        sma20 = close.rolling(20).mean()
        days_in_consolidation = 0
        if len(sma20.dropna()) > 0:
            consolidation_mask = abs(close - sma20) / sma20 < 0.10
            days_in_consolidation = consolidation_mask.iloc[-60:].sum()

            min_days = SOLIDITY_CONFIG['min_consolidation_days']
            if days_in_consolidation >= min_days * 2:
                result['duration_score'] = 0.20
            elif days_in_consolidation >= min_days * 1.5:
                result['duration_score'] = 0.15
            elif days_in_consolidation >= min_days:
                result['duration_score'] = 0.10
            else:
                result['duration_score'] = 0.0

        # Calculate Total
        total_score = (
            result['consolidation_quality'] +
            result['volume_decline'] +
            result['institutional_flow'] +
            result['duration_score']
        )

        result['solidity_score'] = total_score
        result['is_solid'] = total_score >= SOLIDITY_CONFIG['base_threshold']

        # Generate Explanation
        if result['is_solid']:
            components = []
            if result['consolidation_quality'] >= 0.20:
                components.append(f"Tight base ({price_range*100:.1f}% range)")
            if result['volume_decline'] >= 0.15:
                components.append("Declining volume")
            if result['institutional_flow'] >= 0.15:
                components.append(f"DP ${dp_total/1e6:.1f}M")
            if result['duration_score'] >= 0.10:
                components.append(f"{days_in_consolidation}d consolidation")

            result['explanation'] = f"SOLID BASE: Score={total_score:.2f} | " + ", ".join(components)
        else:
            weak = []
            if result['consolidation_quality'] < 0.15:
                weak.append(f"Wide range ({price_range*100:.0f}%)")
            if result['volume_decline'] < 0.10:
                weak.append("No volume decline")
            if result['institutional_flow'] < 0.10:
                weak.append("Low DP activity")

            result['explanation'] = f"Not solid ({total_score:.2f}): " + ", ".join(weak) if weak else f"Sub-threshold ({total_score:.2f})"

    except Exception as e:
        result['explanation'] = f'Solidity analysis error: {str(e)[:50]}'

    return result


def detect_phoenix_reversal(ticker, history_df, full_df=None, dp_support_levels=None, signature_prints=None):
    """
    Flow-Adjusted Dynamic Scoring Model for Phoenix Detection.

    Args:
        ticker: Stock ticker symbol
        history_df: DataFrame with OHLCV data
        full_df: DataFrame with ticker data
        dp_support_levels: Dict mapping tickers to list of DP support prices
        signature_prints: Dict mapping tickers to signature print data

    Returns:
        dict with is_phoenix, phoenix_score, base_duration, volume_ratio, rsi, explanation
    """
    result = {
        'is_phoenix': False,
        'phoenix_score': 0.0,
        'base_duration': 0,
        'volume_ratio': 0.0,
        'rsi': 0.0,
        'explanation': ''
    }

    min_days = PHOENIX_CONFIG['min_base_days']
    max_days = PHOENIX_CONFIG['max_base_days']

    if history_df is None or len(history_df) < min_days:
        result['explanation'] = 'Insufficient history for phoenix analysis'
        return result

    try:
        close = history_df['Close']
        volume = history_df['Volume']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        current_price = float(close.iloc[-1])

        lookback_days = min(len(close), max_days)
        base_period = close.iloc[-lookback_days:]
        base_high = base_period.max()
        base_low = base_period.min()
        base_range = (base_high - base_low) / base_low

        # 52-week high/low
        week_52_period = min(len(close), 252)
        high_52w = close.iloc[-week_52_period:].max()
        low_52w = close.iloc[-week_52_period:].min()
        pct_from_52w_high = (high_52w - current_price) / high_52w if high_52w > 0 else 0
        pct_from_52w_low = (current_price - low_52w) / low_52w if low_52w > 0 else 0
        result['high_52w'] = float(high_52w)
        result['low_52w'] = float(low_52w)
        result['pct_from_52w_high'] = float(pct_from_52w_high)
        result['pct_from_52w_low'] = float(pct_from_52w_low)

        sma50 = close.rolling(50).mean()
        full_lookback = min(len(close), max_days)
        full_sma = sma50.iloc[-full_lookback:] if len(sma50) >= full_lookback else sma50
        full_close = close.iloc[-full_lookback:] if len(close) >= full_lookback else close

        # Dynamic consolidation threshold
        base_consolidation_pct = PHOENIX_CONFIG['min_consolidation_pct']

        avg_volume_50d_prelim = volume.iloc[-50:].mean() if len(volume) >= 50 else volume.mean()
        avg_volume_recent_prelim = volume.iloc[-5:].mean()
        volume_ratio_prelim = avg_volume_recent_prelim / (avg_volume_50d_prelim + 1)
        flow_factor_prelim, _ = calculate_flow_factor(ticker, volume_ratio_prelim, full_df, signature_prints)

        drawdown_bonus = min(0.10, base_range * 0.15)
        flow_bonus_threshold = flow_factor_prelim * 0.05

        consolidation_threshold = base_consolidation_pct + drawdown_bonus + flow_bonus_threshold
        consolidation_threshold = min(consolidation_threshold, 0.30)
        result['consolidation_threshold'] = consolidation_threshold

        days_in_base = ((abs(full_close - full_sma) / full_sma) < consolidation_threshold).sum()
        result['base_duration'] = int(days_in_base)

        # Debug output
        if ENABLE_VALIDATION_MODE and ticker in (VALIDATION_SUITE['institutional_phoenix'] + VALIDATION_SUITE['speculative_phoenix']):
            print(f"  [PHOENIX DEBUG] {ticker}: lookback={full_lookback}d, threshold={consolidation_threshold:.1%}, days_in_base={days_in_base}")

        # Volume ratio
        avg_volume_50d = volume.iloc[-50:].mean() if len(volume) >= 50 else volume.mean()
        avg_volume_recent = volume.iloc[-5:].mean()
        volume_ratio = avg_volume_recent / (avg_volume_50d + 1)
        result['volume_ratio'] = float(volume_ratio)

        has_volume_surge = volume_ratio >= PHOENIX_CONFIG['volume_surge_threshold']

        flow_factor, flow_details = calculate_flow_factor(ticker, volume_ratio, full_df, signature_prints)
        result['flow_factor'] = flow_factor

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        result['rsi'] = current_rsi

        # Dynamic RSI thresholds
        rsi_min_dynamic = max(40, PHOENIX_CONFIG['rsi_min'] - flow_factor * 10)
        rsi_max_dynamic = min(85, PHOENIX_CONFIG['rsi_max'] + flow_factor * 15)
        rsi_in_range = rsi_min_dynamic <= current_rsi <= rsi_max_dynamic
        result['rsi_range'] = f"{rsi_min_dynamic:.0f}-{rsi_max_dynamic:.0f}"

        # Breakout check
        breakout_threshold = PHOENIX_CONFIG['breakout_threshold']
        near_breakout = current_price >= base_low * (1 + base_range * 0.7)
        recent_breakout = (current_price - base_low) / base_low >= breakout_threshold

        # Drawdown constraint
        max_drawdown = (base_high - base_low) / base_high
        acceptable_drawdown = max_drawdown <= PHOENIX_CONFIG['max_drawdown_pct']

        # Dark pool support
        has_dp_support = False
        dp_strength = 0.0
        if dp_support_levels and ticker in dp_support_levels:
            dp_levels = dp_support_levels[ticker]
            nearby_support = [p for p in dp_levels if abs(p - current_price) / current_price < 0.15]
            has_dp_support = len(nearby_support) > 0
            dp_strength = min(len(nearby_support) / 3.0, 1.0)

        # --- MULTI-LAYER SCORING ---

        # Layer 1: Base Duration Score (0-25 points)
        base_duration_score = 0.0
        institutional_threshold = PHOENIX_CONFIG.get('institutional_threshold', 365)

        if min_days <= days_in_base <= max_days:
            if 90 <= days_in_base <= 180:
                base_duration_score = 0.25
            elif 60 <= days_in_base < 90:
                base_duration_score = 0.15 + (days_in_base - 60) / 30 * 0.10
            elif 180 < days_in_base < institutional_threshold:
                base_duration_score = 0.20
            elif institutional_threshold <= days_in_base <= 730:
                if 365 <= days_in_base <= 550:
                    base_duration_score = 0.25
                elif 550 < days_in_base <= 730:
                    base_duration_score = 0.23
        else:
            base_duration_score = 0.0

        # Layer 2: Volume Confirmation Score (0-25 points)
        volume_score = 0.0
        if volume_ratio >= 1.5:
            if volume_ratio >= 3.0:
                volume_score = 0.25
            elif volume_ratio >= 2.0:
                volume_score = 0.20
            else:
                volume_score = 0.10 + (volume_ratio - 1.5) / 0.5 * 0.10
        else:
            if volume_ratio >= 1.2:
                volume_score = 0.05

        # Layer 3: RSI Health Score (0-20 points)
        rsi_score = 0.0
        sweet_spot_min = max(45, 55 - flow_factor * 10)
        sweet_spot_max = min(80, 65 + flow_factor * 15)

        if sweet_spot_min <= current_rsi <= sweet_spot_max:
            if 55 <= current_rsi <= 65:
                rsi_score = 0.20
            elif 50 <= current_rsi <= 70:
                rsi_score = 0.18
            else:
                rsi_score = 0.15 + flow_factor * 0.05
        elif rsi_min_dynamic <= current_rsi <= rsi_max_dynamic:
            if current_rsi < sweet_spot_min:
                rsi_score = 0.08 + flow_factor * 0.04
            else:
                distance_from_70 = current_rsi - 70
                penalty = min(0.10, distance_from_70 * 0.01)
                forgiveness = flow_factor * 0.12
                rsi_score = max(0.05, 0.15 - penalty + forgiveness)
        elif 70 < current_rsi <= 85 and flow_factor >= 0.3:
            rsi_score = 0.05 + flow_factor * 0.08

        result['rsi_sweet_spot'] = f"{sweet_spot_min:.0f}-{sweet_spot_max:.0f}"

        # Layer 4: Breakout Confirmation Score (0-15 points)
        breakout_score = 0.0
        if near_breakout or recent_breakout:
            if recent_breakout:
                breakout_pct = (current_price - base_low) / base_low
                if breakout_pct >= 0.10:
                    breakout_score = 0.15
                elif breakout_pct >= 0.05:
                    breakout_score = 0.12
                else:
                    breakout_score = 0.08
            elif near_breakout:
                breakout_score = 0.10

        # Layer 5: Drawdown Quality Score (0-10 points)
        drawdown_score = 0.0
        if max_drawdown <= 0.70:
            if max_drawdown <= 0.20:
                drawdown_score = 0.10
            elif max_drawdown <= 0.35:
                drawdown_score = 0.08
            elif max_drawdown <= 0.50:
                drawdown_score = 0.07
            elif max_drawdown <= 0.70:
                if days_in_base >= institutional_threshold:
                    drawdown_score = 0.10
                else:
                    drawdown_score = 0.05
        else:
            drawdown_score = -0.05

        # Layer 6: Dark Pool Support Score (0-15 points)
        dp_score = 0.0
        if has_dp_support:
            dp_score = 0.10 + (dp_strength * 0.05)

        # Mega-print bonus
        if full_df is not None and not full_df.empty:
            ticker_rows = full_df[full_df['ticker'] == ticker]
            if not ticker_rows.empty:
                ticker_row = ticker_rows.iloc[0]
                if 'dp_total' in ticker_row:
                    dp_total = ticker_row['dp_total']
                    if dp_total > 50_000_000:
                        mega_print_bonus = min(0.15, math.log10(dp_total / 50_000_000) * 0.10)
                        dp_score += mega_print_bonus
                        if dp_total > 500_000_000:
                            dp_score += 0.10

        # Layer 7: Pattern Synergy Bonus
        synergy_score = 0.0

        # Layer 8: Flow Factor Bonus (0-15 points)
        flow_bonus = 0.0
        if flow_factor >= 0.7:
            flow_bonus = 0.15
        elif flow_factor >= 0.5:
            flow_bonus = 0.10
        elif flow_factor >= 0.3:
            flow_bonus = 0.05

        # Layer 9: Solidity Score (0-18 points)
        solidity_result = calculate_solidity_score(ticker, history_df, full_df, dp_support_levels)
        solidity_score_raw = solidity_result.get('solidity_score', 0)

        solidity_weight = SOLIDITY_CONFIG['weight_in_phoenix']
        solidity_contribution = solidity_score_raw * solidity_weight

        result['solidity_score'] = solidity_score_raw
        result['solidity_details'] = solidity_result

        # --- COMPOSITE SCORE ---
        composite_score = (
            base_duration_score +
            volume_score +
            rsi_score +
            breakout_score +
            drawdown_score +
            dp_score +
            synergy_score +
            flow_bonus +
            solidity_contribution
        )

        # Dynamic threshold
        base_threshold = 0.60
        flow_adjusted_threshold = max(0.45, base_threshold - flow_factor * 0.15)
        is_phoenix = composite_score >= flow_adjusted_threshold

        result['is_phoenix'] = is_phoenix
        result['phoenix_score'] = composite_score
        result['threshold'] = flow_adjusted_threshold

        # --- EXPLANATION ---
        score_components = []

        if is_phoenix:
            if base_duration_score >= 0.20:
                score_components.append(f"{days_in_base}d base ({base_duration_score*100:.0f} pts)")
            if volume_score >= 0.15:
                score_components.append(f"{volume_ratio:.1f}x volume ({volume_score*100:.0f} pts)")
            if rsi_score >= 0.10:
                if current_rsi > 70:
                    score_components.append(f"RSI {current_rsi:.0f} [flow-adjusted] ({rsi_score*100:.0f} pts)")
                else:
                    score_components.append(f"RSI {current_rsi:.0f} ({rsi_score*100:.0f} pts)")
            if breakout_score >= 0.10:
                score_components.append(f"Breakout ({breakout_score*100:.0f} pts)")
            if dp_score >= 0.10:
                score_components.append(f"DP ${flow_details['raw_dp_total']/1e6:.0f}M ({dp_score*100:.0f} pts)")
            if flow_bonus >= 0.05:
                score_components.append(f"Flow Factor {flow_factor:.2f} (+{flow_bonus*100:.0f} pts)")
            if solidity_contribution >= 0.05:
                score_components.append(f"SOLID BASE ({solidity_score_raw:.2f})")

            threshold_note = f" [threshold={flow_adjusted_threshold:.2f}]" if flow_adjusted_threshold < 0.60 else ""
            result['explanation'] = f"PHOENIX REVERSAL: Score={composite_score:.2f}{threshold_note} | " + ", ".join(score_components)
        else:
            weak_components = []
            if base_duration_score < 0.15:
                weak_components.append(f"base {days_in_base}d ({base_duration_score*100:.0f} pts)")
            if volume_score < 0.10:
                weak_components.append(f"volume {volume_ratio:.1f}x ({volume_score*100:.0f} pts)")
            if rsi_score < 0.10:
                rsi_note = f" [range:{rsi_min_dynamic:.0f}-{rsi_max_dynamic:.0f}]" if flow_factor >= 0.2 else ""
                weak_components.append(f"RSI {current_rsi:.0f}{rsi_note} ({rsi_score*100:.0f} pts)")
            if breakout_score < 0.08:
                weak_components.append(f"no breakout ({breakout_score*100:.0f} pts)")

            flow_note = f", flow={flow_factor:.2f}" if flow_factor >= 0.2 else ""
            threshold_note = f"/{flow_adjusted_threshold:.2f}" if flow_adjusted_threshold < 0.60 else "/0.60"

            result['explanation'] = f'Near-phoenix (Score={composite_score:.2f}{threshold_note}{flow_note}): ' + ', '.join(weak_components) if weak_components else f'Sub-threshold pattern (score={composite_score:.2f})'

    except Exception as e:
        result['explanation'] = f'Phoenix detection error: {str(e)[:50]}'

    return result


def detect_cup_and_handle(history_df):
    """
    Detect Cup-and-Handle pattern: U-shaped recovery + small consolidation.

    Args:
        history_df: DataFrame with OHLCV data

    Returns:
        dict with is_cup_handle, cup_handle_score, explanation
    """
    result = {
        'is_cup_handle': False,
        'cup_handle_score': 0.0,
        'explanation': ''
    }

    if history_df is None or len(history_df) < 60:
        result['explanation'] = 'Insufficient history'
        return result

    try:
        close = history_df['Close']
        volume = history_df['Volume']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        lookback = min(len(close), 60)
        cup_period = close.iloc[-lookback:]

        left_rim_high = cup_period.iloc[:20].max()
        cup_low = cup_period.iloc[10:40].min()
        right_rim_high = cup_period.iloc[-20:].max()

        current_price = float(close.iloc[-1])

        cup_depth = (left_rim_high - cup_low) / left_rim_high
        recovery = (right_rim_high - cup_low) / (left_rim_high - cup_low) if left_rim_high > cup_low else 0

        handle_period = close.iloc[-15:]
        handle_high = handle_period.max()
        handle_low = handle_period.min()
        handle_depth = (handle_high - handle_low) / handle_high

        volume_handle = volume.iloc[-15:]
        volume_pre_handle = volume.iloc[-30:-15]
        volume_declining = volume_handle.mean() < volume_pre_handle.mean()

        score = 0.0

        # Cup quality (0-40 points)
        if 0.15 <= cup_depth <= 0.50 and recovery >= 0.80:
            score += 0.40
        elif 0.10 <= cup_depth <= 0.55 and recovery >= 0.70:
            score += 0.25

        # Handle quality (0-30 points)
        if 0.05 <= handle_depth <= 0.15 and volume_declining:
            score += 0.30
        elif 0.05 <= handle_depth <= 0.20:
            score += 0.15

        # Breakout confirmation (0-30 points)
        if current_price >= handle_high * 1.02:
            score += 0.30
        elif current_price >= handle_high * 0.98:
            score += 0.15

        result['is_cup_handle'] = score >= 0.60
        result['cup_handle_score'] = score
        result['explanation'] = f"Cup-Handle: {score:.2f} | Depth={cup_depth*100:.0f}%, Recovery={recovery*100:.0f}%, Handle={handle_depth*100:.0f}%" if score >= 0.60 else f"Not cup-handle (score={score:.2f})"

    except Exception as e:
        result['explanation'] = f'Cup-handle error: {str(e)[:50]}'

    return result


def detect_double_bottom(history_df):
    """
    Detect Double Bottom pattern: Two distinct lows at similar price levels.

    Args:
        history_df: DataFrame with OHLCV data

    Returns:
        dict with is_double_bottom, double_bottom_score, explanation
    """
    result = {
        'is_double_bottom': False,
        'double_bottom_score': 0.0,
        'explanation': ''
    }

    if history_df is None or len(history_df) < 60:
        result['explanation'] = 'Insufficient history'
        return result

    try:
        close = history_df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        lookback = min(len(close), 60)
        pattern_period = close.iloc[-lookback:]

        from scipy.signal import argrelextrema
        local_min_indices = argrelextrema(pattern_period.values, np.less, order=5)[0]

        if len(local_min_indices) < 2:
            result['explanation'] = 'Insufficient local minima'
            return result

        recent_lows = local_min_indices[-2:]
        first_low_idx = recent_lows[0]
        second_low_idx = recent_lows[1]

        first_low = pattern_period.iloc[first_low_idx]
        second_low = pattern_period.iloc[second_low_idx]

        days_between = second_low_idx - first_low_idx

        between_period = pattern_period.iloc[first_low_idx:second_low_idx]
        resistance_high = between_period.max()

        bounce_height = (resistance_high - first_low) / first_low
        low_similarity = abs(second_low - first_low) / first_low

        current_price = float(close.iloc[-1])

        score = 0.0

        # Low similarity (0-35 points)
        if low_similarity <= 0.02:
            score += 0.35
        elif low_similarity <= 0.05:
            score += 0.25
        elif low_similarity <= 0.08:
            score += 0.15

        # Time spacing (0-25 points)
        if 20 <= days_between <= 60:
            score += 0.25
        elif 15 <= days_between < 20 or 60 < days_between <= 80:
            score += 0.15

        # Bounce quality (0-20 points)
        if bounce_height >= 0.15:
            score += 0.20
        elif bounce_height >= 0.10:
            score += 0.15

        # Breakout confirmation (0-20 points)
        if current_price >= resistance_high * 1.03:
            score += 0.20
        elif current_price >= resistance_high:
            score += 0.10

        result['is_double_bottom'] = score >= 0.60
        result['double_bottom_score'] = score
        result['explanation'] = f"Double-Bottom: {score:.2f} | Lows={low_similarity*100:.1f}% apart, Bounce={bounce_height*100:.0f}%, Days={days_between}" if score >= 0.60 else f"Not double-bottom (score={score:.2f})"

    except Exception as e:
        result['explanation'] = f'Double-bottom error: {str(e)[:50]}'

    return result


def apply_smart_gatekeeper(df, config=None):
    """
    Smart Gatekeeper - Pre-filter candidates by dollar-volume liquidity.

    Args:
        df: DataFrame with ticker data
        config: Optional config override (defaults to GATEKEEPER_CONFIG)

    Returns:
        Filtered DataFrame with only liquid/DP-bypassed candidates
    """
    if config is None:
        config = GATEKEEPER_CONFIG

    if df.empty:
        return df

    print("\n[GATEKEEPER v11.0] Applying dollar-volume liquidity filter...")
    original_count = len(df)

    df = df.copy()

    if 'current_price' in df.columns and 'avg_volume' in df.columns:
        df['dollar_volume'] = df['current_price'] * df['avg_volume']
    elif 'current_price' in df.columns:
        df['dollar_volume'] = df['current_price'] * 1_000_000
    else:
        print("  [GATEKEEPER] Warning: Missing price/volume data, skipping filter")
        return df

    def get_liquidity_threshold(row):
        market_cap = row.get('market_cap', 0)
        if market_cap >= config['large_cap_min']:
            return config['large_cap_threshold']
        elif market_cap >= config['mid_cap_min']:
            return config['mid_cap_threshold']
        else:
            return config['small_cap_threshold']

    def has_dp_bypass(row):
        dp_total = row.get('dp_total', 0)
        return dp_total >= config['dp_bypass_threshold']

    passed_liquidity = []
    passed_dp_bypass = []
    rejected = []

    for idx, row in df.iterrows():
        ticker = row.get('ticker', 'Unknown')
        dollar_vol = row.get('dollar_volume', 0)
        threshold = get_liquidity_threshold(row)

        if has_dp_bypass(row):
            passed_dp_bypass.append(idx)
        elif dollar_vol >= threshold:
            passed_liquidity.append(idx)
        else:
            rejected.append(ticker)

    passed_indices = passed_liquidity + passed_dp_bypass
    filtered_df = df.loc[passed_indices].copy()

    liquidity_passed = len(passed_liquidity)
    dp_bypassed = len(passed_dp_bypass)
    rejected_count = len(rejected)

    print(f"  [GATEKEEPER] Passed liquidity filter: {liquidity_passed}")
    print(f"  [GATEKEEPER] DP bypass (institutional): {dp_bypassed}")
    print(f"  [GATEKEEPER] Rejected (illiquid): {rejected_count}")

    if rejected_count > 0 and rejected_count <= 10:
        print(f"  [GATEKEEPER] Rejected tickers: {', '.join(rejected[:10])}")
    elif rejected_count > 10:
        print(f"  [GATEKEEPER] Sample rejected: {', '.join(rejected[:5])}... (+{rejected_count-5} more)")

    return filtered_df


# =============================================================================
# SMART MONEY CONCEPTS (v12) - ICT-style patterns for institutional order flow
# =============================================================================

def detect_choch(history_df, lookback=20):
    """
    Detect Change of Character (CHoCH) - market structure shift.

    CHoCH occurs when:
    - Bullish CHoCH: Price breaks above the last lower high in a downtrend
    - Bearish CHoCH: Price breaks below the last higher low in an uptrend

    Args:
        history_df: DataFrame with OHLCV data
        lookback: Number of bars to analyze for structure

    Returns:
        dict with choch_bullish, choch_bearish, structure_break_level
    """
    result = {
        'choch_bullish': False,
        'choch_bearish': False,
        'structure_break_level': 0.0,
        'choch_score': 0.0
    }

    if history_df is None or len(history_df) < lookback:
        return result

    try:
        close = history_df['Close'].values
        high = history_df['High'].values
        low = history_df['Low'].values

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0].values
            high = high.iloc[:, 0].values
            low = low.iloc[:, 0].values

        # Find swing points in lookback window
        window = min(lookback, len(close) - 1)
        recent_close = close[-window:]
        recent_high = high[-window:]
        recent_low = low[-window:]

        # Identify swing highs and lows (simple 3-bar swing detection)
        swing_highs = []
        swing_lows = []

        for i in range(1, len(recent_high) - 1):
            if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]:
                swing_highs.append((i, recent_high[i]))
            if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]:
                swing_lows.append((i, recent_low[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result

        # Check for downtrend (lower highs) then break above last lower high
        last_high_val = swing_highs[-1][1]
        prev_high_val = swing_highs[-2][1]
        current_close = close[-1]

        if prev_high_val > last_high_val:  # Was in downtrend (lower highs)
            if current_close > last_high_val:  # Broke above last lower high
                result['choch_bullish'] = True
                result['structure_break_level'] = float(last_high_val)
                result['choch_score'] = min(1.0, (current_close - last_high_val) / (last_high_val * 0.02))

        # Check for uptrend (higher lows) then break below last higher low
        last_low_val = swing_lows[-1][1]
        prev_low_val = swing_lows[-2][1]

        if prev_low_val < last_low_val:  # Was in uptrend (higher lows)
            if current_close < last_low_val:  # Broke below last higher low
                result['choch_bearish'] = True
                result['structure_break_level'] = float(last_low_val)
                result['choch_score'] = min(1.0, (last_low_val - current_close) / (last_low_val * 0.02))

    except Exception:
        pass

    return result


def detect_order_blocks(history_df, lookback=30, min_move_pct=0.03):
    """
    Detect Order Blocks - institutional supply/demand zones.

    Bullish OB: Last down candle before a significant up move
    Bearish OB: Last up candle before a significant down move

    Args:
        history_df: DataFrame with OHLCV data
        lookback: Number of bars to search
        min_move_pct: Minimum % move to qualify as significant

    Returns:
        dict with bullish_ob_zone, bearish_ob_zone, ob_score
    """
    result = {
        'bullish_ob_zone': None,  # (low, high) of the order block
        'bearish_ob_zone': None,
        'near_bullish_ob': False,  # Price is near a bullish OB
        'near_bearish_ob': False,
        'ob_score': 0.0  # Strength score
    }

    if history_df is None or len(history_df) < lookback:
        return result

    try:
        close = history_df['Close']
        open_p = history_df['Open']
        high = history_df['High']
        low = history_df['Low']

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            open_p = open_p.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]

        close_arr = close.values
        open_arr = open_p.values
        high_arr = high.values
        low_arr = low.values

        current_price = close_arr[-1]

        # Find bullish order blocks (last red candle before big green move)
        for i in range(len(close_arr) - lookback, len(close_arr) - 3):
            if i < 0:
                continue

            # Check if this is a down candle
            is_down = close_arr[i] < open_arr[i]
            if not is_down:
                continue

            # Check for significant upward move after
            future_high = max(high_arr[i+1:min(i+10, len(high_arr))])
            move_pct = (future_high - close_arr[i]) / close_arr[i]

            if move_pct >= min_move_pct:
                ob_low = low_arr[i]
                ob_high = high_arr[i]

                # Check if price is near this OB
                if ob_low <= current_price <= ob_high * 1.02:
                    result['bullish_ob_zone'] = (float(ob_low), float(ob_high))
                    result['near_bullish_ob'] = True
                    result['ob_score'] = min(1.0, move_pct / 0.05)
                    break

        # Find bearish order blocks (last green candle before big red move)
        for i in range(len(close_arr) - lookback, len(close_arr) - 3):
            if i < 0:
                continue

            is_up = close_arr[i] > open_arr[i]
            if not is_up:
                continue

            future_low = min(low_arr[i+1:min(i+10, len(low_arr))])
            move_pct = (close_arr[i] - future_low) / close_arr[i]

            if move_pct >= min_move_pct:
                ob_low = low_arr[i]
                ob_high = high_arr[i]

                if ob_low * 0.98 <= current_price <= ob_high:
                    result['bearish_ob_zone'] = (float(ob_low), float(ob_high))
                    result['near_bearish_ob'] = True
                    result['ob_score'] = min(1.0, move_pct / 0.05)
                    break

    except Exception:
        pass

    return result


def detect_fvg(history_df, lookback=20, min_gap_pct=0.005):
    """
    Detect Fair Value Gaps (FVG) - price inefficiencies.

    Bullish FVG: Gap between candle 1's high and candle 3's low
    Bearish FVG: Gap between candle 1's low and candle 3's high

    Args:
        history_df: DataFrame with OHLCV data
        lookback: Number of bars to search
        min_gap_pct: Minimum gap size as % of price

    Returns:
        dict with bullish_fvg_zone, bearish_fvg_zone, fvg_score
    """
    result = {
        'bullish_fvg_zone': None,  # (gap_low, gap_high)
        'bearish_fvg_zone': None,
        'in_bullish_fvg': False,  # Price is inside a bullish FVG
        'in_bearish_fvg': False,
        'fvg_score': 0.0
    }

    if history_df is None or len(history_df) < lookback + 3:
        return result

    try:
        close = history_df['Close']
        high = history_df['High']
        low = history_df['Low']

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]

        high_arr = high.values
        low_arr = low.values
        current_price = close.values[-1]

        # Find bullish FVG (gap up - candle 3's low > candle 1's high)
        for i in range(len(close) - lookback, len(close) - 2):
            if i < 0:
                continue

            candle1_high = high_arr[i]
            candle3_low = low_arr[i + 2]

            if candle3_low > candle1_high:
                gap_size = candle3_low - candle1_high
                gap_pct = gap_size / candle1_high

                if gap_pct >= min_gap_pct:
                    result['bullish_fvg_zone'] = (float(candle1_high), float(candle3_low))

                    # Check if price is in or near this FVG
                    if candle1_high * 0.99 <= current_price <= candle3_low * 1.01:
                        result['in_bullish_fvg'] = True
                        result['fvg_score'] = min(1.0, gap_pct / 0.02)

        # Find bearish FVG (gap down - candle 3's high < candle 1's low)
        for i in range(len(close) - lookback, len(close) - 2):
            if i < 0:
                continue

            candle1_low = low_arr[i]
            candle3_high = high_arr[i + 2]

            if candle3_high < candle1_low:
                gap_size = candle1_low - candle3_high
                gap_pct = gap_size / candle1_low

                if gap_pct >= min_gap_pct:
                    result['bearish_fvg_zone'] = (float(candle3_high), float(candle1_low))

                    if candle3_high * 0.99 <= current_price <= candle1_low * 1.01:
                        result['in_bearish_fvg'] = True
                        result['fvg_score'] = min(1.0, gap_pct / 0.02)

    except Exception:
        pass

    return result


def detect_smc_patterns(history_df, lookback=30):
    """
    Combined Smart Money Concepts pattern detection.

    Returns aggregated SMC signals for use in scoring.

    Args:
        history_df: DataFrame with OHLCV data
        lookback: Number of bars to analyze

    Returns:
        dict with smc_bullish_score, smc_bearish_score, and individual pattern flags
    """
    result = {
        'smc_bullish_score': 0.0,
        'smc_bearish_score': 0.0,
        'choch_bullish': False,
        'choch_bearish': False,
        'near_bullish_ob': False,
        'near_bearish_ob': False,
        'in_bullish_fvg': False,
        'in_bearish_fvg': False
    }

    if history_df is None or len(history_df) < lookback:
        return result

    # Detect individual patterns
    choch = detect_choch(history_df, lookback=lookback)
    ob = detect_order_blocks(history_df, lookback=lookback)
    fvg = detect_fvg(history_df, lookback=lookback)

    # Aggregate bullish signals
    bullish_score = 0.0
    if choch['choch_bullish']:
        bullish_score += 0.4 * choch['choch_score']
        result['choch_bullish'] = True
    if ob['near_bullish_ob']:
        bullish_score += 0.35 * ob['ob_score']
        result['near_bullish_ob'] = True
    if fvg['in_bullish_fvg']:
        bullish_score += 0.25 * fvg['fvg_score']
        result['in_bullish_fvg'] = True

    # Aggregate bearish signals
    bearish_score = 0.0
    if choch['choch_bearish']:
        bearish_score += 0.4 * choch['choch_score']
        result['choch_bearish'] = True
    if ob['near_bearish_ob']:
        bearish_score += 0.35 * ob['ob_score']
        result['near_bearish_ob'] = True
    if fvg['in_bearish_fvg']:
        bearish_score += 0.25 * fvg['fvg_score']
        result['in_bearish_fvg'] = True

    result['smc_bullish_score'] = min(1.0, bullish_score)
    result['smc_bearish_score'] = min(1.0, bearish_score)

    return result
