# plotting.py

import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import config
import numpy as np
import seaborn as sns  # Make sure seaborn is installed
import matplotlib.ticker as mtick

matplotlib.use("Agg")


def _get_interval_label(interval_str: str) -> str:
    """
    Helper to convert config's interval string (e.g. 'D', 'W', 'M')
    into a more descriptive word for labeling in the legend.
    """
    interval_str = interval_str.upper()
    if interval_str.startswith("D"):
        return "day"
    elif interval_str.startswith("W"):
        return "week"
    elif interval_str.startswith("M"):
        return "month"
    else:
        return "period"  # Fallback for other intervals


def save_phase_plots(phases, df_phases):
    """
    Generate and save plots for all phases in df_phases (resampled or daily).
    Also makes a heatmap of the 6 phases at the end.
    """
    if df_phases.empty:
        logging.warning("No phase data to plot.")
        return

    df_phases_full = df_phases.copy()  # For moving averages

    # Filter by date range only for the final plotting
    if config.START_DATE:
        start_dt = pd.to_datetime(config.START_DATE)
        df_phases = df_phases[df_phases.index >= start_dt]
    if config.END_DATE:
        end_dt = pd.to_datetime(config.END_DATE)
        df_phases = df_phases[df_phases.index <= end_dt]

    if df_phases.empty:
        logging.warning("No phase data within the specified date range.")
        return

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    interval_label = _get_interval_label(config.PHASE_PLOT_INTERVAL)

    # 1) Plot all phases together
    if config.PLOT_PHASES.get("AllPhases", False):
        plt.figure(figsize=(12, 8))
        for phase in phases:
            if phase not in df_phases.columns:
                continue
            phase_series = df_phases[phase].dropna()
            if phase_series.empty:
                continue

            # Start only at first non-zero date
            non_zero_idx = phase_series[phase_series != 0].index
            if non_zero_idx.empty:
                continue
            first_non_zero_date = non_zero_idx[0]

            # Rolling MA from the full data
            full_series = df_phases_full[phase].dropna()
            moving_avg = full_series.rolling(window=10, min_periods=1).mean()

            phase_series = phase_series.loc[first_non_zero_date:]
            moving_avg = moving_avg.loc[first_non_zero_date:]

            plt.plot(phase_series.index, phase_series.values, label=phase)
            plt.plot(moving_avg.index,
                     moving_avg.values,
                     label=f"{phase} (10 {interval_label} MA)",
                     color="red",
                     linestyle="--")

        plt.title("All Phases Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend()
        plt.grid(True)

        filename = os.path.join(config.RESULTS_DIR, "all_phases.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    # 2) Plot each phase separately
    for phase in phases:
        if not config.PLOT_PHASES.get(phase, False):
            continue
        if phase not in df_phases.columns:
            logging.warning(f"Phase '{phase}' not found in columns. Skipping.")
            continue

        phase_series = df_phases[phase].dropna()
        if phase_series.empty:
            continue

        non_zero_idx = phase_series[phase_series != 0].index
        if non_zero_idx.empty:
            continue
        first_non_zero_date = non_zero_idx[0]

        # Full series for MA
        full_series = df_phases_full[phase].dropna()
        moving_avg = full_series.rolling(window=10, min_periods=1).mean()

        phase_series = phase_series.loc[first_non_zero_date:]
        moving_avg = moving_avg.loc[first_non_zero_date:]

        if phase_series.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(phase_series.index, phase_series.values, label=phase)
        plt.plot(
            moving_avg.index,
            moving_avg.values,
            label=f"{phase} (10 {interval_label} MA)",
            color="red",
            linestyle="--"
        )
        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("Date")
        plt.ylabel("% Tickers in Phase")
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend()
        plt.grid(True)

        filename = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    # 3) Correlation Heatmap among phases
    _save_phase_correlation_heatmap(df_phases)


def _save_phase_correlation_heatmap(df_phases):
    phases_of_interest = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
    existing_cols = [c for c in phases_of_interest if c in df_phases.columns]
    if not existing_cols:
        logging.warning("No phase columns found for correlation heatmap.")
        return

    corr = df_phases[existing_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation of Phases")
    plt.tight_layout()

    filename = os.path.join(config.RESULTS_DIR, "phase_correlation_heatmap.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def save_indicator_plots(indicator_name, df):
    """
    Generate and save plots for an indicator DataFrame with one or more columns.
    Each column is plotted separately. 
    """
    if not config.PLOT_INDICATORS.get(indicator_name, False):
        return
    if df.empty:
        logging.warning(f"No data to plot for indicator '{indicator_name}'.")
        return

    if config.START_DATE:
        start_date = pd.to_datetime(config.START_DATE)
        df = df[df.index >= start_date]
    if config.END_DATE:
        end_date = pd.to_datetime(config.END_DATE)
        df = df[df.index <= end_date]

    if df.empty:
        logging.warning(f"No data in range for '{indicator_name}'.")
        return

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    interval_label = _get_interval_label(config.INDICATOR_PLOT_INTERVAL)

    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(col_data.index, col_data.values, label=col)
        plt.title(f"{indicator_name} - {col} ({interval_label.upper()} data)")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)

        fname = os.path.join(config.RESULTS_DIR, f"{indicator_name}_{col}.png")
        plt.savefig(fname, dpi=300)
        plt.close()


def plot_cumulative_returns(cum_df: pd.DataFrame, output_dir=config.RESULTS_DIR):
    """
    Plot cumulative returns for all columns (all ETFs).
    We do NOT limit to 1 year for the line plots.
    Saves to sector_plots/.
    """
    if cum_df.empty:
        logging.warning("No data to plot in plot_cumulative_returns.")
        return

    plt.figure(figsize=(12, 8))
    for col in cum_df.columns:
        plt.plot(cum_df.index, cum_df[col], label=col)

    plt.title("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    spath = os.path.join(output_dir, "sector_plots")
    os.makedirs(spath, exist_ok=True)
    fpath = os.path.join(spath, "cumulative_returns.png")
    plt.savefig(fpath, dpi=300)
    plt.close()


def plot_rolling_correlation(roll_corr_df: pd.DataFrame, output_dir=config.RESULTS_DIR):
    """
    Plot rolling correlation vs. SPY for each ticker. 
    We do NOT limit to 1 year for line plots.
    Saves to sector_plots/.
    """
    if roll_corr_df.empty:
        logging.warning("No data to plot in plot_rolling_correlation.")
        return

    plt.figure(figsize=(12, 8))
    for col in roll_corr_df.columns:
        plt.plot(roll_corr_df.index, roll_corr_df[col], label=col)

    plt.title(f"Rolling {config.ROLLING_WINDOW}-Day Correlation vs. {config.SP500_TICKER}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Correlation")

    spath = os.path.join(output_dir, "sector_plots")
    os.makedirs(spath, exist_ok=True)
    fpath = os.path.join(spath, "rolling_correlation.png")
    plt.savefig(fpath, dpi=300)
    plt.close()


def plot_relative_performance(rel_df: pd.DataFrame, output_dir=config.RESULTS_DIR):
    """
    Plot ratio of each ETF's price to SPY's price. 
    We do NOT limit to 1 year for line plots.
    Saves to sector_plots/.
    """
    if rel_df.empty:
        logging.warning("No data to plot in plot_relative_performance.")
        return

    plt.figure(figsize=(12, 8))
    for col in rel_df.columns:
        plt.plot(rel_df.index, rel_df[col], label=col)

    plt.title(f"Relative Performance vs. {config.SP500_TICKER}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Price / SPY")

    spath = os.path.join(output_dir, "sector_plots")
    os.makedirs(spath, exist_ok=True)
    fpath = os.path.join(spath, "relative_performance.png")
    plt.savefig(fpath, dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir=config.RESULTS_DIR, title="Correlation Heatmap"):
    """
    Create a correlation heatmap for columns in df, typically 1-year daily returns.
    Saves to sector_plots/.
    """
    if df.empty:
        logging.warning("No data for correlation heatmap.")
        return

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    spath = os.path.join(output_dir, "sector_plots")
    os.makedirs(spath, exist_ok=True)
    fpath = os.path.join(spath, "correlation_heatmap.png")
    plt.savefig(fpath, dpi=300)
    plt.close()
