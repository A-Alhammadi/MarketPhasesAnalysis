# phase_analysis.py

import logging
import numpy as np
import pandas as pd
import gc
import threading
from typing import Dict

import config
from perf_utils import measure_time, log_memory_usage
from db_manager import db_pool
import psycopg2.extras

# We store rolling means in a simple FIFO cache
_rolling_means_cache = {}
_rolling_means_order = []
_cache_lock = threading.Lock()

# Performance counters for cache
_rolling_cache_hits = 0
_rolling_cache_misses = 0


def _evict_if_needed():
    while len(_rolling_means_cache) > config.MAX_ROLLING_MEANS_CACHE_SIZE:
        oldest_key = _rolling_means_order.pop(0)
        _rolling_means_cache.pop(oldest_key, None)


def _get_rolling_means(df: pd.DataFrame, ma_short: int, ma_long: int, ticker: str):
    global _rolling_cache_hits, _rolling_cache_misses
    if df.empty:
        return pd.Series(dtype=np.float32), pd.Series(dtype=np.float32)

    key = (ticker, df.index[0], df.index[-1], len(df), ma_short, ma_long)
    with _cache_lock:
        if key in _rolling_means_cache:
            _rolling_cache_hits += 1
            sma_short_series, sma_long_series = _rolling_means_cache[key]
        else:
            _rolling_cache_misses += 1
            sma_short_series = df["Close"].rolling(window=ma_short, min_periods=1).mean()
            sma_long_series = df["Close"].rolling(window=ma_long, min_periods=1).mean()

            _rolling_means_cache[key] = (sma_short_series, sma_long_series)
            _rolling_means_order.append(key)
            _evict_if_needed()

    return sma_short_series, sma_long_series

def log_rolling_cache_stats():
    """Log stats about the rolling means cache usage."""
    logging.info(
        f"[ROLLING CACHE] Hits: {_rolling_cache_hits}, "
        f"Misses: {_rolling_cache_misses}, "
        f"Current size: {len(_rolling_means_cache)}"
    )

@measure_time
def classify_phases(
    data_dict: Dict[str, pd.DataFrame],
    ma_short: int = 50,
    ma_long: int = 200
):
    """
    Classify each ticker's data into phases based on the position of Close
    relative to rolling averages (ma_short, ma_long).
    
    Returns:
      (df_detail, df_phases_pct)
      
      - df_detail: a DataFrame with columns [Ticker, Close, SMA_50, SMA_200, Phase]
        and Date index. One row per ticker-date.
      
      - df_phases_pct: a DataFrame with Date index and columns for each phase's
        percentage (Bullish, Caution, etc.).
    """
    dfs_with_phases = []
    total_tickers = len(data_dict)

    start_date = pd.to_datetime(config.START_DATE) if config.START_DATE else None
    end_date = pd.to_datetime(config.END_DATE) if config.END_DATE else None

    for ticker, df in data_dict.items():
        try:
            if df.empty:
                logging.warning(f"Ticker {ticker} has empty data. Skipping.")
                continue

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            if df.empty:
                logging.warning(f"No data within the specified date range for {ticker}. Skipping.")
                continue

            df.sort_index(inplace=True)
            if "Close" not in df.columns:
                logging.warning(f"Ticker {ticker} is missing 'Close'. Skipping.")
                continue

            df = df[~df.index.duplicated(keep="first")]
            df["Close"] = df["Close"].astype(np.float32, errors="ignore")

            sma_short_series, sma_long_series = _get_rolling_means(df, ma_short, ma_long, ticker)
            df["SMA_50"] = sma_short_series.astype(np.float32, errors="ignore")
            df["SMA_200"] = sma_long_series.astype(np.float32, errors="ignore")
            
            df.dropna(subset=["Close", "SMA_50", "SMA_200"], inplace=True)

            sma_short_vals = df["SMA_50"]
            sma_long_vals = df["SMA_200"]
            price_vals = df["Close"]

            cond_bullish = (
                (sma_short_vals > sma_long_vals) &
                (price_vals > sma_short_vals) &
                (price_vals > sma_long_vals)
            )
            cond_caution = (
                (sma_short_vals > sma_long_vals) &
                (price_vals < sma_short_vals) &
                (price_vals >= sma_long_vals)
            )
            cond_distribution = (
                (sma_short_vals > sma_long_vals) &
                (price_vals < sma_short_vals) &
                (price_vals < sma_long_vals)
            )
            cond_bearish = (
                (sma_short_vals < sma_long_vals) &
                (price_vals < sma_short_vals) &
                (price_vals < sma_long_vals)
            )
            cond_recuperation = (
                (sma_short_vals < sma_long_vals) &
                (price_vals > sma_short_vals) &
                (price_vals <= sma_long_vals)
            )
            cond_accumulation = (
                (sma_short_vals < sma_long_vals) &
                (price_vals > sma_short_vals) &
                (price_vals > sma_long_vals)
            )

            conditions = [
                cond_bullish,
                cond_caution,
                cond_distribution,
                cond_bearish,
                cond_recuperation,
                cond_accumulation,
            ]
            choices = [
                "Bullish",
                "Caution",
                "Distribution",
                "Bearish",
                "Recuperation",
                "Accumulation",
            ]

            df["Phase"] = np.select(conditions, choices, default="Unknown").astype(object)
            df["Ticker"] = ticker

            # Keep the detail columns for later DB insertion
            dfs_with_phases.append(
                df[["Ticker", "Close", "SMA_50", "SMA_200", "Phase"]]
            )

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
            continue

    if not dfs_with_phases:
        logging.warning("No valid data found across all tickers.")
        empty_detail = pd.DataFrame(columns=["Ticker", "Close", "SMA_50", "SMA_200", "Phase"])
        empty_detail.index.name = "Date"
        empty_summary = pd.DataFrame(
            columns=["Date", "Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        ).set_index("Date")
        return empty_detail, empty_summary

    big_df = pd.concat(dfs_with_phases)
    
    # Aggregated percentage of tickers in each phase per date
    phase_counts = big_df.groupby([big_df.index, "Phase"]).size().unstack(fill_value=0)

    phase_columns = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
    for col in phase_columns:
        if col not in phase_counts.columns:
            phase_counts[col] = 0

    # Convert counts to % of total
    phase_pct = phase_counts.div(total_tickers).mul(100)
    phase_pct = phase_pct[phase_columns]
    phase_pct.index.name = "Date"

    gc.collect()

    # Return both the per-ticker detail and the aggregated summary
    return big_df, phase_pct

