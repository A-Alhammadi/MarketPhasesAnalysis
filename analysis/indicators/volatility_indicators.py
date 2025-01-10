# analysis/indicators/volatility_indicators.py

import pandas as pd
import numpy as np
import logging
from typing import Dict
from analysis.indicators.indicator_template import run_indicator, warnings_aggregator
from perf_utils import measure_time

@measure_time
def compute_chaikin_volatility(close_df: pd.DataFrame, volume_df: pd.DataFrame,
                               high_df: pd.DataFrame, low_df: pd.DataFrame,
                               window=10) -> pd.DataFrame:
    """
    Chaikin Volatility measure based on High-Low range.
    """
    if high_df.empty or low_df.empty:
        warnings_aggregator.set_need_high_low_warning()
        return pd.DataFrame(columns=["ChaikinVolatility"])

    range_df = high_df - low_df
    range_ema = range_df.ewm(span=window, adjust=False).mean()

    cvol_df = range_ema.pct_change(periods=window) * 100.0
    chaikin_vol = cvol_df.mean(axis=1, skipna=True)

    return pd.DataFrame({"ChaikinVolatility": chaikin_vol})


@measure_time
def compute_trin(close_df: pd.DataFrame, volume_df: pd.DataFrame,
                 high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    """
    TRIN calculation based on adv/dec and adv/dec volume.
    """
    if close_df.empty or volume_df.empty:
        warnings_aggregator.set_need_close_volume_warning()
        return pd.DataFrame(columns=["TRIN"])

    prev_close_df = close_df.shift(1)
    up_mask = close_df > prev_close_df
    down_mask = close_df < prev_close_df

    adv_count_series = up_mask.sum(axis=1, skipna=True)
    dec_count_series = down_mask.sum(axis=1, skipna=True)

    adv_vol_series = volume_df.where(up_mask, 0).sum(axis=1, skipna=True)
    dec_vol_series = volume_df.where(down_mask, 0).sum(axis=1, skipna=True)

    adv_count_nonzero = adv_count_series.replace(0, np.nan)
    dec_count_nonzero = dec_count_series.replace(0, np.nan)
    adv_vol_nonzero   = adv_vol_series.replace(0, np.nan)
    dec_vol_nonzero   = dec_vol_series.replace(0, np.nan)

    trin = (adv_count_nonzero / dec_count_nonzero) / (adv_vol_nonzero / dec_vol_nonzero)
    return pd.DataFrame({"TRIN": trin})
