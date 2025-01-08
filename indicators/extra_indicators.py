# indicators/extra_indicators.py

import logging
import pandas as pd
import numpy as np
from typing import Dict

def compute_mcclellan(data_dict: Dict[str, pd.DataFrame], fast: int = 19, slow: int = 39) -> pd.DataFrame:
    """
    Computes the McClellan Oscillator and Summation Index.
    The oscillator is essentially (EMA of adv-decl issues) short minus long period.
    Summation is the cumulative total of the oscillator.

    For this example, we assume we have an 'AdvanceDeclineLine' or we compute adv and decl internally.
    But a standard approach uses "advancing issues - declining issues" each day for the entire market.
    We'll approximate using the number of up-tickers minus down-tickers from data_dict.
    """
    # 1) Construct a daily net adv-decl for the entire market
    #    This is similar to how adv_decline.py does it, but here we want up/down count each day:
    all_dates = sorted({date for df in data_dict.values() for date in df.index})

    adv_minus_decl = []
    for date in all_dates:
        advancers = 0
        decliners = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    if pd.isna(prev_close) or pd.isna(curr_close):
                        continue
                    if curr_close > prev_close:
                        advancers += 1
                    elif curr_close < prev_close:
                        decliners += 1
        net = advancers - decliners
        adv_minus_decl.append([date, net])

    df_net = pd.DataFrame(adv_minus_decl, columns=["Date", "NetAdv"])
    df_net.set_index("Date", inplace=True)
    df_net.sort_index(inplace=True)

    # 2) Compute the EMAs for the short (fast) and long (slow) periods
    df_net["EMA_fast"] = df_net["NetAdv"].ewm(span=fast, adjust=False).mean()
    df_net["EMA_slow"] = df_net["NetAdv"].ewm(span=slow, adjust=False).mean()

    # 3) McClellan Oscillator
    df_net["McClellanOsc"] = df_net["EMA_fast"] - df_net["EMA_slow"]

    # 4) McClellan Summation Index - cumulative sum of the Osc
    df_net["McClellanSum"] = df_net["McClellanOsc"].cumsum()

    return df_net[["McClellanOsc", "McClellanSum"]]


def compute_index_of_fear_greed(
    data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    A simplified example of a multi-factor sentiment measure.
    Typically it might combine:
      - Market volatility (e.g., VIX),
      - Put/Call ratio,
      - Breadth,
      - Junk bond demand,
      - Stock price momentum,
      etc.

    For demonstration, we combine a few internal signals (like adv/decl, new highs/lows) in a toy approach.
    """
    # For a real approach, you'd gather these from external sources or your own breadth data:
    # e.g., use 'compute_adv_decline' or 'compute_new_high_low' results, or external data like VIX.

    # We'll approximate with a "fear" component and a "greed" component:
    #  - "fear" might be fraction of declining issues or new lows
    #  - "greed" might be fraction of advancing issues or new highs

    all_dates = sorted({date for df in data_dict.values() for date in df.index})
    results = []
    total_tickers = len(data_dict)

    for date in all_dates:
        adv = 0
        dec = 0
        new_highs = 0
        new_lows = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    if pd.isna(prev_close) or pd.isna(curr_close):
                        continue
                    if curr_close > prev_close:
                        adv += 1
                    elif curr_close < prev_close:
                        dec += 1
                # simplistic: check if today's close is the highest/lowest over last 20 days, for example
                lookback = 20
                lower_idx = max(0, idx - lookback + 1)
                window_data = df.iloc[lower_idx: idx+1]["Close"]
                if not window_data.empty:
                    if df.iloc[idx]["Close"] == window_data.max():
                        new_highs += 1
                    if df.iloc[idx]["Close"] == window_data.min():
                        new_lows += 1

        # fraction
        adv_frac = adv / total_tickers if total_tickers else 0
        dec_frac = dec / total_tickers if total_tickers else 0
        high_frac = new_highs / total_tickers if total_tickers else 0
        low_frac = new_lows / total_tickers if total_tickers else 0

        # A toy "fear" measure: dec_frac + low_frac
        # A toy "greed" measure: adv_frac + high_frac
        # The final index could be greed - fear, normalized to 0-100 scale, etc.

        fear = (dec_frac + low_frac)
        greed = (adv_frac + high_frac)
        # Example: scale to 0-100, where 50 is neutral
        fg_score = (greed - fear + 1) * 50  # so if greed==fear => 50; if greed>fear => > 50

        results.append([date, fg_score])

    df_fg = pd.DataFrame(results, columns=["Date", "FearGreedIndex"])
    df_fg.set_index("Date", inplace=True)
    return df_fg


def compute_trend_intensity_index(
    data_dict: Dict[str, pd.DataFrame], window: int = 30
) -> pd.DataFrame:
    """
    Measures the strength of a price trend by seeing what fraction of bars close 
    above a chosen moving average vs. below in a given window.

    We'll output a single series: the average TII across all tickers each day.
    That means for each day we see how many tickers are "in trend" vs. "not in trend."
    """
    all_dates = sorted({date for df in data_dict.values() for date in df.index})

    # Prepare MAs
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            logging.warning(f"{ticker}: Missing Close.")
            continue
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

    tii_results = []

    for date in all_dates:
        # We'll compute the ratio for each ticker, then average
        in_trend = 0
        valid_count = 0
        for ticker, df in data_dict.items():
            if date in df.index and f"SMA_{window}" in df.columns:
                sma_val = df.at[date, f"SMA_{window}"]
                close_val = df.at[date, "Close"]
                if not pd.isna(sma_val) and not pd.isna(close_val):
                    valid_count += 1
                    # If Close > SMA, consider that as "above" => trend
                    if close_val > sma_val:
                        in_trend += 1
        
        # The fraction for that day:
        if valid_count > 0:
            daily_tii = (in_trend / valid_count) * 100.0
        else:
            daily_tii = np.nan

        tii_results.append([date, daily_tii])

    df_tii = pd.DataFrame(tii_results, columns=["Date", "TrendIntensityIndex"])
    df_tii.set_index("Date", inplace=True)
    return df_tii


def compute_chaikin_volatility(
    data_dict: Dict[str, pd.DataFrame], window: int = 10
) -> pd.DataFrame:
    """
    Chaikin Volatility uses the difference between high/low range (or some variation) 
    and applies a rate of change to an EMA of that range.

    We'll create a single index that is the average ChaikinVol across all tickers.
    """
    all_dates = sorted({date for df in data_dict.values() for date in df.index})

    # For each ticker, we do: 
    #   1) compute typical range = (High - Low)
    #   2) EMA of that range
    #   3) rate of change(EMA) over 'window' days
    # Then average across all tickers each day.

    chaikin_results = []

    # Precompute for each ticker
    ticker_vol = {}
    for ticker, df in data_dict.items():
        if not {"High", "Low"}.issubset(df.columns):
            logging.warning(f"{ticker}: Missing High/Low for Chaikin Vol.")
            ticker_vol[ticker] = pd.Series(dtype=float)
            continue

        df["Range"] = df["High"] - df["Low"]
        df["RangeEMA"] = df["Range"].ewm(span=window, adjust=False).mean()
        # Rate of change over 'window' days: 
        #   (RangeEMA_today - RangeEMA_(today-window)) / RangeEMA_(today-window) * 100
        df["ChaikinVol"] = df["RangeEMA"].pct_change(periods=window) * 100
        ticker_vol[ticker] = df["ChaikinVol"]

    for date in all_dates:
        values = []
        for ticker, series in ticker_vol.items():
            if date in series.index:
                val = series[date]
                if not pd.isna(val):
                    values.append(val)
        if values:
            avg_val = np.mean(values)
        else:
            avg_val = np.nan

        chaikin_results.append([date, avg_val])

    df_cvol = pd.DataFrame(chaikin_results, columns=["Date", "ChaikinVolatility"])
    df_cvol.set_index("Date", inplace=True)
    return df_cvol


def compute_chaikin_money_flow(
    data_dict: Dict[str, pd.DataFrame], window: int = 20
) -> pd.DataFrame:
    """
    Chaikin Money Flow (CMF) is a volume-weighted average of the Accumulation/Distribution (A/D).
    Typically: 
      Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
      Money Flow Volume = Money Flow Multiplier * Volume
      CMF = (Sum(Money Flow Volume) over N days) / (Sum(Volume) over N days)

    We'll create a single "average" CMF across all tickers each day.
    """
    all_dates = sorted({date for df in data_dict.values() for date in df.index})

    # Precompute for each ticker
    ticker_cmf = {}
    for ticker, df in data_dict.items():
        if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
            logging.warning(f"{ticker}: Missing High/Low/Close/Volume for CMF.")
            ticker_cmf[ticker] = pd.Series(dtype=float)
            continue

        # The multiplier
        df["MF_Mult"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
        # Money Flow Volume
        df["MF_Vol"] = df["MF_Mult"] * df["Volume"]
        # Chaikin Money Flow over 'window' days
        # sum(MF_Vol) / sum(Volume) for last N days
        df["CMF"] = df["MF_Vol"].rolling(window=window).sum() / df["Volume"].rolling(window=window).sum()
        ticker_cmf[ticker] = df["CMF"]

    # Now average across tickers each day
    cmf_results = []
    for date in all_dates:
        daily_vals = []
        for ticker, series in ticker_cmf.items():
            if date in series.index:
                val = series[date]
                if not pd.isna(val):
                    daily_vals.append(val)
        if daily_vals:
            avg_cmf = np.mean(daily_vals)
        else:
            avg_cmf = np.nan
        cmf_results.append([date, avg_cmf])

    df_cmf = pd.DataFrame(cmf_results, columns=["Date", "ChaikinMoneyFlow"])
    df_cmf.set_index("Date", inplace=True)
    return df_cmf
