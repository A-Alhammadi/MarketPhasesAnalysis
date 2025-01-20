# main.py

import logging
import os
import sys
import pandas as pd
import matplotlib
import random
import time
import gc
import cProfile
import pstats
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from concurrent.futures import as_completed, ThreadPoolExecutor

import config
from db_manager import DBConnectionManager, db_pool
from perf_utils import measure_time, log_memory_usage
from plotting import save_phase_plots, save_indicator_plots

# Additional imports
from ticker_utils import get_sp500_tickers
from reporting_utils import save_latest_breadth_values, save_phases_breakdown

# Phase classification
from phase_analysis import classify_phases, log_rolling_cache_stats

# Indicators
from analysis.indicators.indicator_template import clear_caches, run_indicator
from analysis.indicators.breadth_indicators import (
    compute_adv_decline,
    compute_adv_decline_volume,
    compute_mcclellan
)
from analysis.indicators.high_low_indicators import (
    compute_new_high_low,
    compute_percent_above_ma
)
from analysis.indicators.money_flow_indicators import (
    compute_index_of_fear_greed,
    compute_chaikin_money_flow
)
from analysis.indicators.trend_indicators import compute_trend_intensity_index
from analysis.indicators.volatility_indicators import compute_chaikin_volatility, compute_trin

matplotlib.use("Agg")  # Headless mode

# db_helpers imports
from db_helpers import (
    create_tables,
    write_phase_details_to_db,
    detect_and_log_changes,
    write_indicator_data_to_db,
    batch_fetch_from_db
)
from analysis.indicators.volume_indicators import (
    compute_and_store_volume_mas,
    compute_and_store_volume_ma_deviation,
    export_extreme_volumes
)
from analysis.indicators.price_indicators import compute_and_store_price_ma_deviation

# Use the updated run_sector_analysis that saves data to sector_analysis
from sector_analysis import run_sector_analysis

# For profiling
profiler = cProfile.Profile()
profiler.enable()

@atexit.register
def stop_profiler():
    profiler.disable()
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    profile_stats_path = os.path.join(config.RESULTS_DIR, "profile_stats.txt")
    with open(profile_stats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats()


@measure_time
def main():
    """
    Main execution flow:
      1) Create tables if not found (including sector_data).
      2) Get list of tickers (e.g., S&P 500).
      3) Fetch existing data from DB for those tickers (price_data).
      4) Compute & store volume MAs (10,20).
      5) Compute & store price % dev (50/200).
      6) Compute & store volume % dev (20/63).
      7) Classify phases & store => also do plots.
      8) Compute other indicators & store => also do plots.
      9) Export "extreme volume" file (uses config.EXTREME_VOLUME_Z_THRESHOLD).
      10) Save "latest breadth" text.
      11) Detect/log changes (phase, SMA crosses) -> writes to config.PHASE_CHANGES_FILE,
          config.PRICE_SMA_CHANGES_FILE.
      12) Sector analysis from 'sector_data' (optional, controlled by config.RUN_SECTOR_ANALYSIS).
    """

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    # Log setup
    log_path = os.path.join(config.RESULTS_DIR, "logs.txt")
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Additional event logger
    events_logger = logging.getLogger("events_logger")
    events_logger.setLevel(logging.INFO)
    events_log_path = os.path.join(config.RESULTS_DIR, "events_log.txt")
    fh = logging.FileHandler(events_log_path, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    events_logger.addHandler(fh)

    events_logger.info("START of main program")

    with DBConnectionManager() as conn:
        if conn is None:
            logging.error("Could not establish a database connection. Exiting.")
            events_logger.error("Could not establish DB connection. Exiting.")
            sys.exit(1)

        db_pool.monitor_pool()

        # 1) Ensure all necessary tables are created
        create_tables(conn)
        events_logger.info("Tables ensured in DB")

        # 2) Tickers (S&P 500 from Wikipedia, for example)
        tickers = get_sp500_tickers()
        if not tickers:
            logging.error("No tickers found. Exiting.")
            events_logger.error("No tickers found. Exiting.")
            sys.exit(1)

        # 3) Fetch data from DB's 'price_data' table:
        log_memory_usage("Before batch_fetch_from_db")
        data_dict = batch_fetch_from_db(tickers, conn)
        db_pool.monitor_pool()
        log_memory_usage("After batch_fetch_from_db")

        if not data_dict:
            logging.warning("No data found in DB for these tickers.")
            events_logger.warning("No data found. Exiting.")
            sys.exit(0)

        # 4) Volume MAs
        compute_and_store_volume_mas(data_dict, conn)

        # 5) Price dev (50, 200)
        compute_and_store_price_ma_deviation(data_dict, conn)

        # 6) Volume dev (20, 63)
        compute_and_store_volume_ma_deviation(data_dict, conn)

        # 7) Classify phases
        df_phases_detail, df_phases_daily = classify_phases(
            data_dict,
            ma_short=config.MA_SHORT,
            ma_long=config.MA_LONG
        )
        log_rolling_cache_stats()

        if df_phases_daily.empty:
            logging.warning("Phase classification is empty. Skipping phase plots.")
            events_logger.warning("No phases data - skipping.")
        else:
            # Write daily phases to an 'indicator_data' style table if desired
            write_indicator_data_to_db(df_phases_daily, "phase_classification", conn)
            write_phase_details_to_db(df_phases_detail, conn)

            # For plotting
            df_phases_daily.index = pd.to_datetime(df_phases_daily.index, errors="coerce")
            df_phases_resampled = df_phases_daily.resample(config.PHASE_PLOT_INTERVAL).last()

            save_phase_plots(
                phases=["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"],
                df_phases=df_phases_resampled
            )
            save_phases_breakdown(df_phases_daily, df_phases_resampled)
            events_logger.info("Phase plots saved")

        # 8) Other indicators
        indicator_tasks = [
            ("adv_decline", compute_adv_decline),
            ("adv_decline_volume", compute_adv_decline_volume),
            ("new_high_low", compute_new_high_low),
            ("percent_above_ma", compute_percent_above_ma),
            ("mcclellan", compute_mcclellan),
            ("fear_greed", compute_index_of_fear_greed),
            ("trend_intensity_index", compute_trend_intensity_index),
            ("chaikin_volatility", compute_chaikin_volatility),
            ("chaikin_money_flow", compute_chaikin_money_flow),
            ("trin", compute_trin),
        ]

        computed_indicators = {}
        for indicator_name, func_ in indicator_tasks:
            events_logger.info(f"Starting computation for {indicator_name}")
            result_df_daily = run_indicator(
                indicator_name=indicator_name,
                data_dict=data_dict,
                compute_func=lambda cdf, vdf, hdf, ldf: func_(cdf, vdf, hdf, ldf)
            )

            if result_df_daily.empty:
                logging.warning(f"{indicator_name} returned empty. Skipping.")
                events_logger.warning(f"{indicator_name} is empty.")
                continue

            # Store results in DB
            write_indicator_data_to_db(result_df_daily, indicator_name, conn)
            events_logger.info(f"Inserted {indicator_name} to DB")

            # Plot
            result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
            if not isinstance(result_df_daily.index, pd.DatetimeIndex) or result_df_daily.index.hasnans:
                logging.warning(f"{indicator_name} index invalid. Skipping plot.")
                continue

            result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
            save_indicator_plots(indicator_name, result_df_resampled)
            events_logger.info(f"Saved {indicator_name} plots")

            computed_indicators[indicator_name] = result_df_daily

        # 9) Extreme volume
        export_extreme_volumes(conn)  # uses config.EXTREME_VOLUME_Z_THRESHOLD internally

        # 10) Save latest breadth
        save_latest_breadth_values(
            df_phases_daily,
            computed_indicators,
            output_file=os.path.join(config.RESULTS_DIR, "breadth_values.txt")
        )
        events_logger.info("Saved latest breadth & phase data")

        # 11) Detect changes => writes to config.PHASE_CHANGES_FILE and config.PRICE_SMA_CHANGES_FILE
        detect_and_log_changes(conn)
        events_logger.info("Logged phase/indicator changes")

        # 12) Sector analysis
        if config.RUN_SECTOR_ANALYSIS:
            events_logger.info("Starting sector analysis from 'sector_data'...")
            run_sector_analysis(conn, config.SECTOR_TICKERS)
            events_logger.info("Sector analysis complete.")
        else:
            events_logger.info("Skipping sector analysis (RUN_SECTOR_ANALYSIS=False).")

        logging.info("All tasks completed successfully.")
        events_logger.info("END of main program")

if __name__ == "__main__":
    main()
