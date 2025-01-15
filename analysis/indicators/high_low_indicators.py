# analysis/indicators/high_low_indicators.py

import pandas as pd
import numpy as np
from typing import Dict
import logging

from analysis.indicators.indicator_template import (
    run_indicator, warnings_aggregator, df_registry
)
from perf_utils import measure_time
import config

@measure_time
def compute_new_high_low(close_df: pd.DataFrame,
                         volume_df: pd.DataFrame,
                         high_df: pd.DataFrame,
                         low_df: pd.DataFrame,
                         lookback=config.NEWHIGHLOW_LOOKBACK) -> pd.DataFrame:
    """
    New High / New Low count.
    The default lookback is now pulled from config.NEWHIGHLOW_LOOKBACK.
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["NewHighCount","NewLowCount","NHNL_Diff","NHNL_Ratio"])

    close_df_id = id(close_df)
    # Rolling max/min can fallback to pandas if needed, but let's do it here for clarity
    rolling_max = close_df.rolling(lookback, min_periods=1).max()
    rolling_min = close_df.rolling(lookback, min_periods=1).min()

    new_high_mask = close_df.eq(rolling_max)
    new_low_mask  = close_df.eq(rolling_min)

    nh_count = new_high_mask.sum(axis=1, skipna=True)
    nl_count = new_low_mask.sum(axis=1, skipna=True)

    nhnl_diff = nh_count - nl_count
    nhnl_ratio = nh_count / nl_count.replace(0, np.nan)

    return pd.DataFrame({
        "NewHighCount": nh_count,
        "NewLowCount": nl_count,
        "NHNL_Diff": nhnl_diff,
        "NHNL_Ratio": nhnl_ratio
    })


@measure_time
def compute_percent_above_ma(close_df: pd.DataFrame,
                             volume_df: pd.DataFrame,
                             high_df: pd.DataFrame,
                             low_df: pd.DataFrame,
                             ma_window=config.PERCENTABOVE_MA_WINDOW) -> pd.DataFrame:
    """
    Percentage of stocks above the specified moving average.
    The default ma_window is now pulled from config.PERCENTABOVE_MA_WINDOW.
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["PercentAboveMA"])

    sma = close_df.rolling(ma_window, min_periods=1).mean()

    above_mask = close_df > sma
    above_count = above_mask.sum(axis=1, skipna=True)
    total_tickers = above_mask.count(axis=1)

    pct_above = (above_count / total_tickers) * 100.0
    return pd.DataFrame({"PercentAboveMA": pct_above})
