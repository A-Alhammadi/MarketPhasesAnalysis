# analysis/indicators/trend_indicators.py

import pandas as pd
import numpy as np
import logging
from typing import Dict
from analysis.indicators.indicator_template import run_indicator, warnings_aggregator
from perf_utils import measure_time

@measure_time
def compute_trend_intensity_index(close_df: pd.DataFrame, volume_df: pd.DataFrame, 
                                  high_df: pd.DataFrame, low_df: pd.DataFrame,
                                  window=30) -> pd.DataFrame:
    """
    Trend Intensity Index based on how many stocks are above their SMA.
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["TrendIntensityIndex"])

    sma = close_df.rolling(window, min_periods=1).mean()

    above_mask = close_df > sma
    fraction_above = above_mask.sum(axis=1, skipna=True) / above_mask.count(axis=1)
    tii = fraction_above * 100.0

    return pd.DataFrame({"TrendIntensityIndex": tii})
