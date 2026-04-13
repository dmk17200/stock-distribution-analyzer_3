"""
Signal Grading Backtester
=========================
Replays the scanner logic on historical data using a rolling window,
records what signal it WOULD have produced on each past date,
then grades each signal against actual forward returns.

Usage:
    python backtest_signals.py                  # Run full backtest
    python backtest_signals.py --tickers AAPL MSFT GOOG
    python backtest_signals.py --warmup 300 --horizons 5 10 21 63

Outputs:
    backtest_results/snapshots.csv              — every signal snapshot
    backtest_results/signal_grade_report.csv    — hit rates & avg returns per signal
    backtest_results/per_signal_component.csv   — individual indicator grading
    backtest_results/ic_timeseries.csv          — daily information coefficient
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import ndimage, stats
from scipy.signal import find_peaks
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
OUTPUT_DIR = "backtest_results"
DEFAULT_HORIZONS = [5, 10, 21, 63]  # trading days forward
DEFAULT_WARMUP = 300                 # days of data needed before first signal
TRAILING_WINDOW = 252                # rolling window for z-score (1 year)
DEFAULT_DAYS = 720                   # total history to fetch


# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_full_history(ticker, days=DEFAULT_DAYS):
    """Fetch daily OHLCV for a ticker. Returns DataFrame or None."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "Close": "price",
            "Volume": "volume", "High": "high", "Low": "low"
        })
        df["stock"] = ticker
        df = df.sort_values("date").reset_index(drop=True)
        return df[["stock", "date", "open", "price", "high", "low", "volume"]]
    except Exception as e:
        print(f"  [WARN] Could not fetch {ticker}: {e}")
        return None


# ============================================================================
# POINT-IN-TIME SIGNAL COMPUTATION
# ============================================================================
# These functions mirror your app's logic but operate on a TRUNCATED slice
# of data — only rows up to and including the simulation date.

def compute_moving_averages(prices, windows=[20, 50, 200]):
    """Compute MAs from a price series. Returns dict of window→value."""
    result = {}
    for w in windows:
        if len(prices) >= w:
            result[w] = prices[-w:].mean()
        else:
            result[w] = np.nan
    return result


def compute_rolling_zscore(prices, volumes, window=TRAILING_WINDOW):
    """
    Volume-weighted z-score using a TRAILING window (not full history).
    This is the corrected version — no lookahead bias.
    """
    if len(prices) < 50:
        return 0.0, np.nan, np.nan

    lookback = min(window, len(prices))
    p = prices[-lookback:]
    v = volumes[-lookback:]

    total_vol = v.sum()
    if total_vol <= 0:
        return 0.0, np.mean(p), np.std(p)

    weighted_mean = np.average(p, weights=v)
    weighted_var = np.average((p - weighted_mean) ** 2, weights=v)
    std_dev = np.sqrt(weighted_var) if weighted_var > 0 else 1.0

    current_price = prices[-1]
    z = (current_price - weighted_mean) / std_dev if std_dev > 0 else 0.0
    return z, weighted_mean, std_dev


def compute_volume_profile_position(prices, volumes):
    """Compute distance from nearest high-volume node (top 20%)."""
    if len(prices) < 50:
        return 0.0, "UNKNOWN", "UNKNOWN", None

    current_price = prices[-1]
    price_rounded = np.round(prices).astype(int)
    vp = pd.Series(volumes, index=price_rounded).groupby(level=0).sum()

    if vp.empty:
        return 0.0, "UNKNOWN", "UNKNOWN", None

    threshold = vp.quantile(0.80)
    hvn = vp[vp >= threshold]
    if hvn.empty:
        return 0.0, "UNKNOWN", "UNKNOWN", None

    nearest_node = min(hvn.index, key=lambda x: abs(float(x) - current_price))
    nearest_node = float(nearest_node)
    pct_from_node = abs(current_price - nearest_node) / nearest_node * 100 if nearest_node != 0 else 0

    # Context
    if pct_from_node < 5:
        context = "AT_VALUE"
    elif pct_from_node > 40:
        context = "EXTREMELY_EXTENDED"
    elif pct_from_node > 15:
        context = "EXTENDED"
    else:
        context = "TRANSITIONAL"

    # Direction
    if current_price > nearest_node * 1.05:
        direction = "ABOVE_VALUE"
    elif current_price < nearest_node * 0.95:
        direction = "BELOW_VALUE"
    else:
        direction = "AT_VALUE"

    return pct_from_node, context, direction, nearest_node


def compute_tsmom(prices, windows=[20, 60, 126, 252]):
    """Time-series momentum across multiple lookback windows."""
    current = prices[-1]
    results = {}
    for w in windows:
        if len(prices) >= w:
            past = prices[-w]
            ret = (current / past) - 1
            results[w] = {"return": ret, "signal": "BUY" if ret > 0 else "SELL"}
        else:
            results[w] = {"return": None, "signal": "N/A"}

    signals = [v["signal"] for v in results.values() if v["signal"] != "N/A"]
    buy_count = signals.count("BUY")
    total = len(signals)
    if total == 0:
        alignment = "INSUFFICIENT_DATA"
    elif buy_count == total:
        alignment = "FULL_BULL"
    elif buy_count == 0:
        alignment = "FULL_BEAR"
    elif buy_count > total - buy_count:
        alignment = "LEANING_BULL"
    elif buy_count < total - buy_count:
        alignment = "LEANING_BEAR"
    else:
        alignment = "MIXED"

    results["alignment"] = alignment
    results["buy_count"] = buy_count
    results["total_windows"] = total
    return results


def detect_parabolic(prices, lookback=20):
    """Detect parabolic acceleration — mirrors app logic."""
    if len(prices) < lookback:
        return False, 0.0
    recent = prices[-lookback:]
    half = lookback // 2
    first_change = (recent[half - 1] - recent[0]) / recent[0] if recent[0] != 0 else 0
    second_change = (recent[-1] - recent[half]) / recent[half] if recent[half] != 0 else 0

    if second_change > first_change * 1.5 and second_change > 0.15:
        return True, second_change
    if second_change < first_change * 1.5 and second_change < -0.15:
        return True, second_change
    return False, 0.0


def compute_composite_signal(prices, volumes):
    """
    Run the full scoring algorithm on a point-in-time slice.
    Returns a dict with signal, net_score, and all sub-components.
    """
    current_price = prices[-1]

    # --- Moving Averages ---
    mas = compute_moving_averages(prices)
    ma_50 = mas.get(50, current_price)
    ma_200 = mas.get(200, current_price)
    if np.isnan(ma_50): ma_50 = current_price
    if np.isnan(ma_200): ma_200 = current_price

    pct_above_ma50 = ((current_price - ma_50) / ma_50) * 100 if ma_50 != 0 else 0
    pct_above_ma200 = ((current_price - ma_200) / ma_200) * 100 if ma_200 != 0 else 0

    if current_price > ma_50 and current_price > ma_200:
        ma_position = "ABOVE_BOTH"
    elif current_price < ma_50 and current_price < ma_200:
        ma_position = "BELOW_BOTH"
    else:
        ma_position = "MIXED"

    # --- Golden / Death Cross ---
    if len(prices) >= 200:
        ma50_series = pd.Series(prices).rolling(50).mean().dropna().values
        ma200_series = pd.Series(prices).rolling(200).mean().dropna().values
        min_len = min(len(ma50_series), len(ma200_series))
        if min_len > 0:
            crossover = "GOLDEN_CROSS" if ma50_series[-1] > ma200_series[-1] else "DEATH_CROSS"
        else:
            crossover = "NEUTRAL"
    else:
        crossover = "NEUTRAL"

    # --- Volume Profile ---
    pct_from_node, vol_context, vol_direction, nearest_node = compute_volume_profile_position(prices, volumes)

    # --- Z-Score (rolling) ---
    z_score, weighted_mean, std_dev = compute_rolling_zscore(prices, volumes)

    # --- TSMOM ---
    tsmom = compute_tsmom(prices)
    momentum_state = tsmom["alignment"]

    # --- Parabolic ---
    is_parabolic, accel_rate = detect_parabolic(prices)

    # ==================================================================
    # SCORING (mirrors app logic exactly)
    # ==================================================================
    bullish = 0
    bearish = 0

    # 1. MA position
    if ma_position == "ABOVE_BOTH":
        bullish += 2
    elif ma_position == "BELOW_BOTH":
        bearish += 2

    # 2. Crossover
    if crossover == "GOLDEN_CROSS":
        bullish += 2
    elif crossover == "DEATH_CROSS":
        bearish += 2

    # 3. Volume profile
    if vol_context == "AT_VALUE":
        if vol_direction in ("AT_VALUE", "BELOW_VALUE"):
            bullish += 2
        elif vol_direction == "ABOVE_VALUE":
            bearish += 1
    elif vol_context == "EXTENDED":
        if vol_direction == "ABOVE_VALUE":
            bearish += 4
        elif vol_direction == "BELOW_VALUE":
            bullish += 3
    elif vol_context == "EXTREMELY_EXTENDED":
        if vol_direction == "ABOVE_VALUE":
            bearish += 6
        elif vol_direction == "BELOW_VALUE":
            bullish += 4

    # 4. Z-score
    if z_score > 10:
        bearish += 6
    elif z_score > 5:
        bearish += 4
    elif z_score > 2:
        bearish += 2
    elif z_score < -10:
        bullish += 6
    elif z_score < -5:
        bullish += 4
    elif z_score < -2:
        bullish += 2

    # 5. Extension from MAs
    if pct_above_ma200 > 50:
        bearish += 4
    elif pct_above_ma200 > 30:
        bearish += 3
    elif pct_above_ma200 > 20 or pct_above_ma50 > 15:
        bearish += 2
    elif pct_above_ma200 > 12 or pct_above_ma50 > 10:
        bearish += 1
    elif pct_above_ma200 < -30:
        bullish += 3
    elif pct_above_ma200 < -20 or pct_above_ma50 < -15:
        bullish += 1

    # 6. Parabolic
    if is_parabolic:
        if accel_rate > 0:
            bearish += 3
        else:
            bullish += 2

    # 7. TSMOM
    if momentum_state == "FULL_BULL":
        bullish += 2
    elif momentum_state == "LEANING_BULL":
        bullish += 1
    elif momentum_state == "LEANING_BEAR":
        bearish += 1
    elif momentum_state == "FULL_BEAR":
        bearish += 2

    # Cross-signals
    if momentum_state == "FULL_BULL" and vol_context in ("EXTENDED", "EXTREMELY_EXTENDED"):
        bearish += 1
    if momentum_state in ("FULL_BEAR", "LEANING_BEAR") and vol_context == "AT_VALUE":
        bearish += 1

    # 8. Extreme conditions
    extreme_bull_count = sum([
        pct_from_node > 40,
        z_score > 5,
        pct_above_ma200 > 40,
        is_parabolic and accel_rate > 0,
    ])
    if extreme_bull_count >= 3:
        bearish += 5

    extreme_bear_count = sum([
        pct_from_node > 20 and (nearest_node is not None and current_price < nearest_node),
        z_score < -4,
        pct_above_ma200 < -25,
    ])
    if extreme_bear_count >= 2:
        bullish += 4

    # --- Signal ---
    net_score = bullish - bearish
    if net_score >= 6:
        signal = "STRONG_BUY"
    elif net_score >= 4:
        signal = "BUY"
    elif net_score >= 2:
        signal = "HOLD_WATCH_FOR_BUY"
    elif net_score <= -6:
        if ma_position == "ABOVE_BOTH" and momentum_state in ("FULL_BULL", "LEANING_BULL"):
            signal = "WAIT_FOR_PULLBACK"
        else:
            signal = "STRONG_SELL"
    elif net_score <= -4:
        if ma_position == "ABOVE_BOTH" and momentum_state in ("FULL_BULL", "LEANING_BULL"):
            signal = "HOLD_WATCH_FOR_SELL"
        else:
            signal = "SELL"
    elif net_score <= -2:
        signal = "HOLD_WATCH_FOR_SELL"
    else:
        signal = "HOLD_NEUTRAL"

    # Overrides
    if extreme_bull_count >= 3:
        if ma_position == "ABOVE_BOTH" and momentum_state in ("FULL_BULL", "LEANING_BULL"):
            signal = "WAIT_FOR_PULLBACK"
        else:
            signal = "STRONG_SELL"
    if vol_context == "EXTENDED" and vol_direction == "ABOVE_VALUE" and net_score > 0:
        signal = "WAIT_FOR_PULLBACK"

    return {
        "signal": signal,
        "net_score": net_score,
        "bullish_score": bullish,
        "bearish_score": bearish,
        "z_score": round(z_score, 3),
        "pct_above_ma50": round(pct_above_ma50, 2),
        "pct_above_ma200": round(pct_above_ma200, 2),
        "ma_position": ma_position,
        "crossover": crossover,
        "vol_context": vol_context,
        "vol_direction": vol_direction,
        "pct_from_node": round(pct_from_node, 2),
        "tsmom_alignment": momentum_state,
        "tsmom_20d": tsmom.get(20, {}).get("return"),
        "tsmom_60d": tsmom.get(60, {}).get("return"),
        "tsmom_126d": tsmom.get(126, {}).get("return"),
        "tsmom_252d": tsmom.get(252, {}).get("return"),
        "is_parabolic": is_parabolic,
        "accel_rate": round(accel_rate, 4) if is_parabolic else None,
    }


# ============================================================================
# BACKTESTING LOOP
# ============================================================================
def backtest_ticker(df, ticker, warmup, horizons, sample_every=1):
    """
    Walk through history for one ticker, compute signals at each date,
    and attach forward returns.

    Parameters
    ----------
    df : DataFrame with columns [date, price, volume, ...] sorted by date
    ticker : str
    warmup : int — skip this many rows before first signal
    horizons : list of int — forward return horizons in trading days
    sample_every : int — compute signal every N days (1 = daily, 5 = weekly)

    Returns
    -------
    List of snapshot dicts
    """
    prices_all = df["price"].values.astype(float)
    volumes_all = df["volume"].values.astype(float)
    dates_all = df["date"].values
    n = len(df)

    snapshots = []
    max_horizon = max(horizons)

    for i in range(warmup, n, sample_every):
        # Truncate to point-in-time
        prices_pit = prices_all[: i + 1]
        volumes_pit = volumes_all[: i + 1]
        sim_date = dates_all[i]
        sim_price = prices_all[i]

        # Compute signal
        try:
            result = compute_composite_signal(prices_pit, volumes_pit)
        except Exception:
            continue

        snapshot = {
            "ticker": ticker,
            "date": sim_date,
            "price": round(sim_price, 2),
        }
        snapshot.update(result)

        # Forward returns
        for h in horizons:
            future_idx = i + h
            if future_idx < n:
                fwd_price = prices_all[future_idx]
                fwd_return = (fwd_price / sim_price) - 1
                snapshot[f"fwd_return_{h}d"] = round(fwd_return, 5)
            else:
                snapshot[f"fwd_return_{h}d"] = np.nan

        # Max favorable / adverse excursion within longest horizon
        future_slice = prices_all[i + 1 : min(i + max_horizon + 1, n)]
        if len(future_slice) > 0:
            snapshot["max_favorable_excursion"] = round(
                (future_slice.max() / sim_price) - 1, 5
            )
            snapshot["max_adverse_excursion"] = round(
                (future_slice.min() / sim_price) - 1, 5
            )
        else:
            snapshot["max_favorable_excursion"] = np.nan
            snapshot["max_adverse_excursion"] = np.nan

        snapshots.append(snapshot)

    return snapshots


# ============================================================================
# GRADING
# ============================================================================
def grade_signals(snap_df, horizons):
    """
    Grade composite signals: hit rate, avg return, sample count per signal.
    """
    reports = []
    signal_col = "signal"
    signals = snap_df[signal_col].unique()

    for sig in sorted(signals):
        subset = snap_df[snap_df[signal_col] == sig]
        row = {"signal": sig, "count": len(subset)}

        is_bullish = sig in ("STRONG_BUY", "BUY", "HOLD_WATCH_FOR_BUY")
        is_bearish = sig in ("STRONG_SELL", "SELL", "HOLD_WATCH_FOR_SELL")

        for h in horizons:
            col = f"fwd_return_{h}d"
            valid = subset[col].dropna()
            if len(valid) == 0:
                row[f"avg_return_{h}d"] = np.nan
                row[f"hit_rate_{h}d"] = np.nan
                row[f"median_return_{h}d"] = np.nan
                continue

            row[f"avg_return_{h}d"] = round(valid.mean() * 100, 3)
            row[f"median_return_{h}d"] = round(valid.median() * 100, 3)

            if is_bullish:
                row[f"hit_rate_{h}d"] = round((valid > 0).mean() * 100, 1)
            elif is_bearish:
                row[f"hit_rate_{h}d"] = round((valid < 0).mean() * 100, 1)
            else:
                # For neutral signals, hit rate = % positive (informational)
                row[f"hit_rate_{h}d"] = round((valid > 0).mean() * 100, 1)

        # MFE / MAE
        mfe = subset["max_favorable_excursion"].dropna()
        mae = subset["max_adverse_excursion"].dropna()
        row["avg_mfe"] = round(mfe.mean() * 100, 2) if len(mfe) > 0 else np.nan
        row["avg_mae"] = round(mae.mean() * 100, 2) if len(mae) > 0 else np.nan

        reports.append(row)

    return pd.DataFrame(reports)


def grade_individual_components(snap_df, horizons):
    """
    Grade each sub-signal independently.
    For continuous signals (z_score, tsmom_252d, etc.): compute IC.
    For categorical signals (ma_position, vol_context, etc.): compute avg return per category.
    """
    results = []
    primary_horizon = 21 if 21 in horizons else horizons[0]
    fwd_col = f"fwd_return_{primary_horizon}d"

    valid = snap_df.dropna(subset=[fwd_col])
    if valid.empty:
        return pd.DataFrame()

    # --- Continuous signals: information coefficient (rank correlation) ---
    continuous_cols = [
        ("net_score", "Composite net score"),
        ("z_score", "Rolling z-score"),
        ("pct_above_ma200", "% above 200-MA"),
        ("pct_above_ma50", "% above 50-MA"),
        ("pct_from_node", "% from VP node"),
        ("tsmom_20d", "TSMOM 20d return"),
        ("tsmom_60d", "TSMOM 60d return"),
        ("tsmom_126d", "TSMOM 126d return"),
        ("tsmom_252d", "TSMOM 252d return"),
    ]

    for col, label in continuous_cols:
        if col not in valid.columns:
            continue
        sub = valid[[col, fwd_col]].dropna()
        if len(sub) < 30:
            continue
        rho, pval = spearmanr(sub[col], sub[fwd_col])
        # Also compute directional hit rate for the signal's implied direction
        # Positive score → expect positive return
        if col in ("z_score", "pct_above_ma200", "pct_above_ma50", "pct_from_node"):
            # These are CONTRARIAN: high value → expect negative return
            correct = ((sub[col] > 0) & (sub[fwd_col] < 0)) | ((sub[col] < 0) & (sub[fwd_col] > 0))
        else:
            # These are MOMENTUM: high value → expect positive return
            correct = ((sub[col] > 0) & (sub[fwd_col] > 0)) | ((sub[col] < 0) & (sub[fwd_col] < 0))

        results.append({
            "component": label,
            "type": "continuous",
            "IC": round(rho, 4),
            "IC_pvalue": round(pval, 4),
            "directional_hit_rate": round(correct.mean() * 100, 1),
            "n_obs": len(sub),
            "horizon": f"{primary_horizon}d",
        })

    # --- Categorical signals: avg return per category ---
    categorical_cols = [
        ("ma_position", "MA position"),
        ("crossover", "Golden/Death cross"),
        ("vol_context", "Volume context"),
        ("vol_direction", "Volume direction"),
        ("tsmom_alignment", "TSMOM alignment"),
        ("is_parabolic", "Parabolic flag"),
    ]

    for col, label in categorical_cols:
        if col not in valid.columns:
            continue
        grouped = valid.groupby(col)[fwd_col].agg(["mean", "median", "count"])
        for cat, row in grouped.iterrows():
            if row["count"] < 10:
                continue
            cat_subset = valid[valid[col] == cat][fwd_col]
            hit = (cat_subset > 0).mean()
            results.append({
                "component": f"{label} = {cat}",
                "type": "categorical",
                "avg_return_pct": round(row["mean"] * 100, 3),
                "median_return_pct": round(row["median"] * 100, 3),
                "hit_rate_up": round(hit * 100, 1),
                "n_obs": int(row["count"]),
                "horizon": f"{primary_horizon}d",
            })

    return pd.DataFrame(results)


def compute_daily_ic(snap_df, horizons):
    """
    Compute cross-sectional IC (Spearman rank correlation between net_score
    and forward return) for each date that has >= 5 stocks.
    """
    records = []
    primary_horizon = 21 if 21 in horizons else horizons[0]
    fwd_col = f"fwd_return_{primary_horizon}d"

    for date, group in snap_df.groupby("date"):
        valid = group[["net_score", fwd_col]].dropna()
        if len(valid) < 5:
            continue
        rho, _ = spearmanr(valid["net_score"], valid[fwd_col])
        records.append({
            "date": date,
            "IC": round(rho, 4),
            "n_stocks": len(valid),
        })

    ic_df = pd.DataFrame(records)
    if not ic_df.empty:
        ic_df["IC_cumulative"] = ic_df["IC"].expanding().mean().round(4)
    return ic_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Backtest signal grading")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Specific tickers to test (default: load from portfolio CSV)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup days before first signal (default: {DEFAULT_WARMUP})")
    parser.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS,
                        help=f"Forward return horizons (default: {DEFAULT_HORIZONS})")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Compute signal every N days (1=daily, 5=weekly). Default: 5")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Days of history to fetch (default: {DEFAULT_DAYS})")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Determine tickers ---
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        # Try to load from portfolio CSV
        portfolio_file = "Robinhood_December_1_-_Sheet1.csv"
        possible_paths = [portfolio_file, f"data/{portfolio_file}"]
        tickers = []
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    pdf = pd.read_csv(path)
                    tickers = pdf["Symbol"].str.strip().tolist()
                    print(f"Loaded {len(tickers)} tickers from {path}")
                except Exception:
                    pass
                break
        if not tickers:
            tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA",
                        "NVDA", "SPY", "QQQ", "JPM"]
            print(f"No portfolio file found. Using default tickers: {tickers}")

    # Always include SPY for cross-sectional IC
    if "SPY" not in tickers:
        tickers.append("SPY")

    print(f"\n{'='*60}")
    print(f"SIGNAL GRADING BACKTEST")
    print(f"{'='*60}")
    print(f"Tickers:       {len(tickers)}")
    print(f"Warmup:        {args.warmup} days")
    print(f"Horizons:      {args.horizons} days")
    print(f"Sample every:  {args.sample_every} days")
    print(f"History:        {args.days} days")
    print(f"Output:        {OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    # --- Fetch data ---
    all_snapshots = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {ticker}...", end=" ")
        df = fetch_full_history(ticker, days=args.days)
        if df is None or len(df) < args.warmup + max(args.horizons):
            print(f"SKIP (insufficient data: {len(df) if df is not None else 0} rows)")
            continue

        snapshots = backtest_ticker(
            df, ticker,
            warmup=args.warmup,
            horizons=args.horizons,
            sample_every=args.sample_every,
        )
        all_snapshots.extend(snapshots)
        print(f"OK — {len(snapshots)} snapshots")

    if not all_snapshots:
        print("\n[ERROR] No snapshots generated. Check data availability.")
        sys.exit(1)

    snap_df = pd.DataFrame(all_snapshots)
    snap_df["date"] = pd.to_datetime(snap_df["date"])

    # --- Save raw snapshots ---
    snap_path = os.path.join(OUTPUT_DIR, "snapshots.csv")
    snap_df.to_csv(snap_path, index=False)
    print(f"\n✅ Saved {len(snap_df)} snapshots → {snap_path}")

    # --- Grade composite signals ---
    print("\nGrading composite signals...")
    grade_df = grade_signals(snap_df, args.horizons)
    grade_path = os.path.join(OUTPUT_DIR, "signal_grade_report.csv")
    grade_df.to_csv(grade_path, index=False)
    print(f"✅ Saved signal grades → {grade_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SIGNAL GRADE SUMMARY")
    print(f"{'='*60}")
    print(grade_df.to_string(index=False))

    # --- Grade individual components ---
    print("\nGrading individual signal components...")
    comp_df = grade_individual_components(snap_df, args.horizons)
    comp_path = os.path.join(OUTPUT_DIR, "per_signal_component.csv")
    if not comp_df.empty:
        comp_df.to_csv(comp_path, index=False)
        print(f"✅ Saved component grades → {comp_path}")

        # Print continuous ICs
        continuous = comp_df[comp_df["type"] == "continuous"].copy()
        if not continuous.empty:
            print(f"\n{'='*60}")
            print("INFORMATION COEFFICIENTS (rank correlation with forward return)")
            print(f"{'='*60}")
            display_cols = ["component", "IC", "IC_pvalue", "directional_hit_rate", "n_obs"]
            avail = [c for c in display_cols if c in continuous.columns]
            print(continuous[avail].to_string(index=False))
    else:
        print("  [WARN] Not enough data for component grading")

    # --- Daily cross-sectional IC ---
    print("\nComputing daily cross-sectional IC...")
    ic_df = compute_daily_ic(snap_df, args.horizons)
    if not ic_df.empty:
        ic_path = os.path.join(OUTPUT_DIR, "ic_timeseries.csv")
        ic_df.to_csv(ic_path, index=False)
        print(f"✅ Saved IC timeseries → {ic_path}")

        avg_ic = ic_df["IC"].mean()
        median_ic = ic_df["IC"].median()
        pct_positive = (ic_df["IC"] > 0).mean() * 100
        print(f"\n   Average IC:     {avg_ic:.4f}")
        print(f"   Median IC:      {median_ic:.4f}")
        print(f"   % days IC > 0:  {pct_positive:.1f}%")
        print(f"   (IC > 0.05 is significant, > 0.10 is strong)")
    else:
        print("  [WARN] Not enough cross-sectional data for IC calculation")

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")
    print(f"Total snapshots:     {len(snap_df)}")
    print(f"Unique tickers:      {snap_df['ticker'].nunique()}")
    print(f"Date range:          {snap_df['date'].min().date()} → {snap_df['date'].max().date()}")
    print(f"\nFiles saved in:      {OUTPUT_DIR}/")
    print(f"  snapshots.csv              — raw signal + forward return data")
    print(f"  signal_grade_report.csv    — hit rates & avg returns per signal")
    print(f"  per_signal_component.csv   — individual indicator IC & grading")
    print(f"  ic_timeseries.csv          — daily cross-sectional IC\n")

    # --- Key takeaways ---
    if not grade_df.empty:
        print(f"{'='*60}")
        print("KEY TAKEAWAYS")
        print(f"{'='*60}")

        for _, row in grade_df.iterrows():
            sig = row["signal"]
            n = row["count"]
            primary_h = 21 if 21 in args.horizons else args.horizons[0]
            hr_col = f"hit_rate_{primary_h}d"
            ar_col = f"avg_return_{primary_h}d"
            hr = row.get(hr_col, np.nan)
            ar = row.get(ar_col, np.nan)

            if pd.isna(hr) or n < 10:
                status = "⚪ INSUFFICIENT DATA"
            elif sig in ("STRONG_BUY", "BUY", "HOLD_WATCH_FOR_BUY"):
                if hr >= 55:
                    status = "✅ KEEP (hit rate ≥ 55%)"
                elif hr >= 50:
                    status = "🟡 MARGINAL (50-55%)"
                else:
                    status = "❌ FAILING (< 50% — worse than coin flip)"
            elif sig in ("STRONG_SELL", "SELL", "HOLD_WATCH_FOR_SELL"):
                if hr >= 55:
                    status = "✅ KEEP (hit rate ≥ 55%)"
                elif hr >= 50:
                    status = "🟡 MARGINAL (50-55%)"
                else:
                    status = "❌ FAILING (< 50% — worse than coin flip)"
            else:
                status = f"⚪ NEUTRAL (avg return: {ar:+.2f}%)" if not pd.isna(ar) else "⚪ NEUTRAL"

            print(f"  {sig:30s}  n={n:4d}  hit={hr:5.1f}%  avg={ar:+6.2f}%  → {status}"
                  if not pd.isna(hr) else f"  {sig:30s}  n={n:4d}  → {status}")


if __name__ == "__main__":
    main()