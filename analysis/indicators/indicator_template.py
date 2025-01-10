# analysis/indicators/indicator_template.py

import logging
import pandas as pd
import numpy as np
import gc
import threading

from typing import Dict, Callable
from db_manager import DBConnectionManager
from perf_utils import measure_time
import config

###############################################################################
# Thread-safe aggregator for warnings
###############################################################################
class WarningAggregator:
    """
    Thread-safe aggregator for warnings and missing columns.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.missing_close_tickers = []
        self.missing_volume_tickers = []
        self.missing_high_tickers = []
        self.missing_low_tickers = []
        self.need_close_warning = False
        self.need_close_volume_warning = False
        self.need_high_low_warning = False

    def add_missing_close_ticker(self, ticker: str):
        with self._lock:
            self.missing_close_tickers.append(ticker)

    def add_missing_volume_ticker(self, ticker: str):
        with self._lock:
            self.missing_volume_tickers.append(ticker)

    def add_missing_high_ticker(self, ticker: str):
        with self._lock:
            self.missing_high_tickers.append(ticker)

    def add_missing_low_ticker(self, ticker: str):
        with self._lock:
            self.missing_low_tickers.append(ticker)

    def set_need_close_warning(self, value: bool = True):
        with self._lock:
            self.need_close_warning = value

    def set_need_close_volume_warning(self, value: bool = True):
        with self._lock:
            self.need_close_volume_warning = value

    def set_need_high_low_warning(self, value: bool = True):
        with self._lock:
            self.need_high_low_warning = value

    def get_warnings(self):
        with self._lock:
            return {
                "missing_close_tickers": list(self.missing_close_tickers),
                "missing_volume_tickers": list(self.missing_volume_tickers),
                "missing_high_tickers": list(self.missing_high_tickers),
                "missing_low_tickers": list(self.missing_low_tickers),
                "need_close_warning": self.need_close_warning,
                "need_close_volume_warning": self.need_close_volume_warning,
                "need_high_low_warning": self.need_high_low_warning
            }

    def clear(self):
        with self._lock:
            self.missing_close_tickers.clear()
            self.missing_volume_tickers.clear()
            self.missing_high_tickers.clear()
            self.missing_low_tickers.clear()
            self.need_close_warning = False
            self.need_close_volume_warning = False
            self.need_high_low_warning = False


warnings_aggregator = WarningAggregator()

###############################################################################
# DataFrame registry (for large data memory management)
###############################################################################
import weakref

class DataFrameRegistry:
    """
    Weak-reference-based registry for DataFrames with a simple FIFO eviction.
    Dynamically set its size from config.
    """
    def __init__(self, maxsize: int = 1000):
        self._registry = {}
        self._maxsize = maxsize
        self._insertion_order = []
        self._lock = threading.Lock()

    def register(self, df: pd.DataFrame) -> int:
        df_id = id(df)
        with self._lock:
            if len(self._registry) >= self._maxsize:
                oldest_id = self._insertion_order.pop(0)
                if oldest_id in self._registry:
                    del self._registry[oldest_id]

            self._registry[df_id] = weakref.ref(df, self._cleanup)
            self._insertion_order.append(df_id)
        return df_id

    def get(self, df_id: int):
        with self._lock:
            ref = self._registry.get(df_id)
            if ref is not None:
                df = ref()
                if df is not None:
                    return df
                # If the df is already garbage-collected, remove from registry
                self._cleanup(ref)
        return None

    def _cleanup(self, weak_ref):
        with self._lock:
            to_remove = []
            for k, v in self._registry.items():
                if v() is None:
                    to_remove.append(k)

            for k in to_remove:
                self._registry.pop(k, None)
                if k in self._insertion_order:
                    self._insertion_order.remove(k)

    def clear(self):
        with self._lock:
            self._registry.clear()
            self._insertion_order.clear()


df_registry = DataFrameRegistry(maxsize=config.MAX_ROLLING_MEANS_CACHE_SIZE)

###############################################################################
# Helper functions
###############################################################################
def preprocess_ticker_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logging.error("Error in preprocess_ticker_df: %s", e)
        raise


def gather_price_dfs(data_dict: Dict[str, pd.DataFrame]) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Gather close, volume, high, low DataFrames from a dict of ticker->DataFrame.
    Register them in df_registry if not empty.
    """
    close_map, volume_map = {}, {}
    high_map, low_map = {}, {}

    for ticker, df in data_dict.items():
        try:
            df = preprocess_ticker_df(df)
        except Exception:
            continue

        if "Close" not in df.columns:
            warnings_aggregator.add_missing_close_ticker(ticker)
            continue
        else:
            close_map[ticker] = df["Close"]

        if "Volume" in df.columns:
            volume_map[ticker] = df["Volume"]
        else:
            warnings_aggregator.add_missing_volume_ticker(ticker)

        if "High" in df.columns:
            high_map[ticker] = df["High"]
        else:
            warnings_aggregator.add_missing_high_ticker(ticker)

        if "Low" in df.columns:
            low_map[ticker] = df["Low"]
        else:
            warnings_aggregator.add_missing_low_ticker(ticker)

    close_df = pd.DataFrame(close_map).sort_index() if close_map else pd.DataFrame()
    volume_df = pd.DataFrame(volume_map).sort_index() if volume_map else pd.DataFrame()
    high_df = pd.DataFrame(high_map).sort_index() if high_map else pd.DataFrame()
    low_df = pd.DataFrame(low_map).sort_index() if low_map else pd.DataFrame()

    # Register them
    if not close_df.empty:
        df_registry.register(close_df)
    if not volume_df.empty:
        df_registry.register(volume_df)
    if not high_df.empty:
        df_registry.register(high_df)
    if not low_df.empty:
        df_registry.register(low_df)

    return close_df, volume_df, high_df, low_df


def log_warning_summary():
    w = warnings_aggregator.get_warnings()

    if w["missing_close_tickers"]:
        logging.warning(
            "Missing 'Close' column for tickers (skipped): %s",
            ", ".join(w["missing_close_tickers"])
        )
    if w["missing_volume_tickers"]:
        logging.warning(
            "Missing 'Volume' column for tickers (NaN in volume_df): %s",
            ", ".join(w["missing_volume_tickers"])
        )
    if w["missing_high_tickers"]:
        logging.warning(
            "Missing 'High' column for tickers (NaN in high_df): %s",
            ", ".join(w["missing_high_tickers"])
        )
    if w["missing_low_tickers"]:
        logging.warning(
            "Missing 'Low' column for tickers (NaN in low_df): %s",
            ", ".join(w["missing_low_tickers"])
        )
    if w["need_close_warning"]:
        logging.warning("No close data found where required.")
    if w["need_close_volume_warning"]:
        logging.warning("Need close & volume data for some indicators.")
    if w["need_high_low_warning"]:
        logging.warning("Need High/Low data for some indicators.")


def clear_caches():
    """
    Clear caches (rolling results, registry, warnings) periodically.
    """
    warnings_aggregator.clear()
    df_registry.clear()
    gc.collect()


###############################################################################
# INDICATOR TEMPLATE
###############################################################################
def run_indicator(
    indicator_name: str,
    data_dict: Dict[str, pd.DataFrame],
    compute_func: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Standard pattern for running an indicator. Gathers the necessary DataFrames (close, volume, etc.)
    and calls compute_func to do the actual calculation. Then logs warnings if needed.
    """
    close_df, volume_df, high_df, low_df = gather_price_dfs(data_dict)
    result = compute_func(close_df, volume_df, high_df, low_df)
    log_warning_summary()
    return result
