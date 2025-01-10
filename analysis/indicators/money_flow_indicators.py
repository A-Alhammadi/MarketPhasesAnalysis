# analysis/indicators/money_flow_indicators.py

import pandas as pd
import numpy as np
import logging
from typing import Dict
from analysis.indicators.indicator_template import run_indicator, warnings_aggregator
from perf_utils import measure_time

@measure_time
def compute_index_of_fear_greed(close_df: pd.DataFrame, volume_df: pd.DataFrame, 
                                high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    """
    A custom Fear-Greed Index measure, combining adv/dec fraction with new highs/lows.
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["FearGreedIndex"])

    prev_close = close_df.shift(1)
    up_mask = close_df > prev_close
    down_mask = close_df < prev_close

    adv_count = up_mask.sum(axis=1, skipna=True)
    dec_count = down_mask.sum(axis=1, skipna=True)
    total_count = close_df.count(axis=1)

    adv_frac = adv_count / total_count.replace(0, np.nan)
    dec_frac = dec_count / total_count.replace(0, np.nan)

    lookback = 20
    rolling_max_20 = close_df.rolling(lookback, min_periods=1).max()
    rolling_min_20 = close_df.rolling(lookback, min_periods=1).min()

    new_high_mask = close_df.eq(rolling_max_20)
    new_low_mask  = close_df.eq(rolling_min_20)

    nh_count = new_high_mask.sum(axis=1, skipna=True)
    nl_count = new_low_mask.sum(axis=1, skipna=True)

    new_high_frac = nh_count / total_count.replace(0, np.nan)
    new_low_frac  = nl_count / total_count.replace(0, np.nan)

    fear = dec_frac + new_low_frac
    greed = adv_frac + new_high_frac

    fg_score = (greed - fear + 1) * 50.0

    return pd.DataFrame({"FearGreedIndex": fg_score})


@measure_time
def compute_chaikin_money_flow(close_df: pd.DataFrame, volume_df: pd.DataFrame,
                               high_df: pd.DataFrame, low_df: pd.DataFrame, 
                               window=20) -> pd.DataFrame:
    """
    Chaikin Money Flow indicator.
    """
    if close_df.empty or volume_df.empty or high_df.empty or low_df.empty:
        if close_df.empty or volume_df.empty:
            warnings_aggregator.set_need_close_volume_warning()
        if high_df.empty or low_df.empty:
            warnings_aggregator.set_need_high_low_warning()
        return pd.DataFrame(columns=["ChaikinMoneyFlow"])

    range_ = (high_df - low_df).replace(0, np.nan)
    mf_mult = ((close_df - low_df) - (high_df - close_df)) / range_
    mf_vol  = mf_mult * volume_df

    sum_mf_vol = mf_vol.rolling(window, min_periods=1).sum()
    sum_vol    = volume_df.rolling(window, min_periods=1).sum()

    cmf_df = sum_mf_vol / sum_vol
    cmf_series = cmf_df.mean(axis=1, skipna=True)

    return pd.DataFrame({"ChaikinMoneyFlow": cmf_series})
