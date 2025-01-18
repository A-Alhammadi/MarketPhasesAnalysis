# sector_analysis.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

import config
from db_helpers import (
    batch_fetch_sector_data,
    write_sector_analysis_metrics
)
from analysis.calculations import (
    calculate_daily_returns,
    calculate_rolling_correlation,
    calculate_relative_performance,
    calculate_cumulative_returns
)
from plotting import (
    plot_cumulative_returns,
    plot_rolling_correlation,
    plot_relative_performance,
    plot_correlation_heatmap
)

def run_sector_analysis(conn, etf_tickers: List[str]):
    """
    1) Fetch data from 'sector_data' for these ETFs (including SPY).
    2) Compute daily_return, cumulative_return, rolling_corr_spy, relative_perf_spy.
    3) Store those metrics in 'sector_analysis' table.
    4) Generate:
        - Full-range line plots for cumulative returns, rolling correlation, relative perf
        - A 1-year correlation heatmap only
    """

    if not etf_tickers:
        logging.warning("No ETF tickers provided. Skipping sector analysis.")
        return

    # 1) Fetch data from sector_data
    data_dict = batch_fetch_sector_data(etf_tickers, conn)
    if not data_dict:
        logging.warning("No data returned for these ETFs. Skipping analysis.")
        return

    # Build a wide DataFrame of close prices
    close_map = {}
    for ticker, df_ticker in data_dict.items():
        if df_ticker.empty or "Close" not in df_ticker.columns:
            logging.warning(f"No valid Close data for ticker: {ticker}. Skipping.")
            continue
        close_map[ticker] = df_ticker["Close"]

    if not close_map:
        logging.warning("No valid close data for sector ETF analysis. Skipping.")
        return

    close_df = pd.DataFrame(close_map).sort_index()

    # Drop columns that are entirely NaN
    close_df.dropna(how="all", axis=1, inplace=True)

    if close_df.empty:
        logging.warning("All tickers had empty or NaN data. Nothing to plot.")
        return

    # 2) Calculations
    daily_returns = calculate_daily_returns(close_df)
    cumulative_ret_df = calculate_cumulative_returns(daily_returns)

    rolling_corr_df = pd.DataFrame()
    relative_perf_df = pd.DataFrame()

    if config.SP500_TICKER in close_df.columns:
        rolling_corr_df = calculate_rolling_correlation(daily_returns, benchmark=config.SP500_TICKER)
        relative_perf_df = calculate_relative_performance(close_df, benchmark=config.SP500_TICKER)
    else:
        logging.warning(f"SP500_TICKER {config.SP500_TICKER} not found in columns. No correlation/relative perf vs SPY.")

    # 3) Store them in 'sector_analysis' DB
    records = []
    for date_ in close_df.index:
        data_date = date_.date()
        for ticker in close_df.columns:
            dr = daily_returns.at[date_, ticker]
            cr = cumulative_ret_df.at[date_, ticker]

            # rolling_corr vs SPY
            rc_spy = rolling_corr_df.at[date_, ticker] if (not rolling_corr_df.empty and ticker in rolling_corr_df) else None
            # relative perf vs SPY
            rp_spy = relative_perf_df.at[date_, ticker] if (not relative_perf_df.empty and ticker in relative_perf_df) else None

            # Convert non-NaN to float, else None
            dr = float(dr) if pd.notna(dr) else None
            cr = float(cr) if pd.notna(cr) else None
            rc_spy = float(rc_spy) if pd.notna(rc_spy) else None
            rp_spy = float(rp_spy) if pd.notna(rp_spy) else None

            records.append((data_date, ticker, dr, cr, rc_spy, rp_spy))

    if not records:
        logging.warning("No valid records to insert into sector_analysis.")
        return
    write_sector_analysis_metrics(conn, records)
    logging.info("Inserted sector analysis metrics into DB.")

    # 4) Plots
    # Full-range line plots
    plot_cumulative_returns(cumulative_ret_df, output_dir=config.RESULTS_DIR)
    if not rolling_corr_df.empty:
        plot_rolling_correlation(rolling_corr_df, output_dir=config.RESULTS_DIR)
    if not relative_perf_df.empty:
        plot_relative_performance(relative_perf_df, output_dir=config.RESULTS_DIR)

    # 1-year correlation heatmap for daily returns
    if not daily_returns.empty:
        one_year_ago = daily_returns.index.max() - pd.DateOffset(days=252)
        daily_returns_1yr = daily_returns[daily_returns.index >= one_year_ago]
        if not daily_returns_1yr.empty:
            plot_correlation_heatmap(
                daily_returns_1yr,
                output_dir=config.RESULTS_DIR,
                title="1-Year Daily Returns Correlation Heatmap"
            )
        else:
            logging.warning("1-year daily returns is empty; no heatmap produced.")
    else:
        logging.warning("daily_returns is empty; no correlation heatmap produced.")

    logging.info("Sector analysis complete: plots created, data saved.")
