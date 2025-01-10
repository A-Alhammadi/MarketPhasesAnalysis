# analysis/indicators/breadth_indicators.py

import pandas as pd
import numpy as np
import logging
from typing import Dict
from analysis.indicators.indicator_template import run_indicator, warnings_aggregator
from perf_utils import measure_time


@measure_time
def compute_adv_decline(close_df: pd.DataFrame, volume_df: pd.DataFrame, 
                        high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    """
    Advance-Decline line based on close_df. 
    Ignores volume_df, high_df, low_df in this specific calculation.
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["AdvanceDeclineLine"])

    prev_close_df = close_df.shift(1)
    up_mask = close_df > prev_close_df
    down_mask = close_df < prev_close_df

    advancers_per_day = up_mask.sum(axis=1, skipna=True)
    decliners_per_day = down_mask.sum(axis=1, skipna=True)

    net_adv = advancers_per_day - decliners_per_day
    ad_line = net_adv.cumsum()

    return pd.DataFrame({"AdvanceDeclineLine": ad_line})


@measure_time
def compute_adv_decline_volume(close_df: pd.DataFrame, volume_df: pd.DataFrame, 
                               high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    """
    Advance-Decline Volume line. Uses close_df & volume_df.
    """
    if close_df.empty or volume_df.empty:
        warnings_aggregator.set_need_close_volume_warning()
        return pd.DataFrame(columns=["AdvDeclVolumeLine"])

    prev_close_df = close_df.shift(1)
    up_mask = close_df > prev_close_df
    down_mask = close_df < prev_close_df

    up_volume = volume_df.where(up_mask, 0)
    down_volume = volume_df.where(down_mask, 0)

    net_volume = up_volume.sum(axis=1, skipna=True) - down_volume.sum(axis=1, skipna=True)
    cum_vol_line = net_volume.cumsum()

    return pd.DataFrame({"AdvDeclVolumeLine": cum_vol_line})


@measure_time
def compute_mcclellan(close_df: pd.DataFrame, volume_df: pd.DataFrame, 
                      high_df: pd.DataFrame, low_df: pd.DataFrame, fast=19, slow=39) -> pd.DataFrame:
    """
    McClellan Oscillator & Summation Index (breadth measure).
    """
    if close_df.empty:
        warnings_aggregator.set_need_close_warning()
        return pd.DataFrame(columns=["McClellanOsc","McClellanSum"])

    prev_close = close_df.shift(1)
    up_mask = close_df > prev_close
    down_mask = close_df < prev_close

    adv_count = up_mask.sum(axis=1, skipna=True)
    dec_count = down_mask.sum(axis=1, skipna=True)
    net_adv = adv_count - dec_count

    df_net = pd.DataFrame({"NetAdv": net_adv})
    df_net["EMA_fast"] = df_net["NetAdv"].ewm(span=fast, adjust=False).mean()
    df_net["EMA_slow"] = df_net["NetAdv"].ewm(span=slow, adjust=False).mean()

    df_net["McClellanOsc"] = df_net["EMA_fast"] - df_net["EMA_slow"]
    df_net["McClellanSum"] = df_net["McClellanOsc"].cumsum()

    return df_net[["McClellanOsc", "McClellanSum"]]
