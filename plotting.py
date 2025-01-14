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

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import pandas as pd
import config

def save_phase_plots(phases, df_phases):
    """
    Generate and save plots for all phases in df_phases (resampled or daily).
    Ensures consistent y-axis (0% to 100%), overlays a red 10-(interval) MA,
    and starts plotting only from the first non-zero date to avoid empty portion.

    Also creates a heatmap correlation plot of the 6 phases at the end.
    """

    if df_phases.empty:
        logging.warning("No phase data to plot.")
        return

    # Filter by date range
    if config.START_DATE:
        start_date = pd.to_datetime(config.START_DATE)
        df_phases_full = df_phases.copy()  # Keep the full data for MA calculations
        df_phases = df_phases[df_phases.index >= start_date]
    if config.END_DATE:
        end_date = pd.to_datetime(config.END_DATE)
        df_phases = df_phases[df_phases.index <= end_date]

    if df_phases.empty:
        logging.warning("No phase data within the specified date range to plot.")
        return

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    interval_label = _get_interval_label(config.PHASE_PLOT_INTERVAL)

    # ----------------------------
    # 1) Plot all phases together
    # ----------------------------
    if config.PLOT_PHASES.get("AllPhases", False):
        plt.figure(figsize=(12, 8))

        for phase in phases:
            if phase not in df_phases.columns:
                continue

            # Drop NaN first
            phase_series = df_phases[phase].dropna()

            # Start only at the first non-zero date
            non_zero_indices = phase_series[phase_series != 0].index
            if non_zero_indices.empty:
                # If the entire series is zeros (or empty after dropna),
                # skip this phase
                continue
            first_non_zero_date = non_zero_indices[0]

            # Include data before the first non-zero date for MA calculation
            phase_series_full = df_phases_full[phase].dropna()
            moving_avg = phase_series_full.rolling(window=10, min_periods=1).mean()

            # Trim to start from the first non-zero date for plotting
            phase_series = phase_series.loc[first_non_zero_date:]
            moving_avg = moving_avg.loc[first_non_zero_date:]

            # Plot a line only (no markers)
            plt.plot(phase_series.index, phase_series.values, label=phase)

            # Plot the moving average line
            plt.plot(
                moving_avg.index,
                moving_avg.values,
                label=f"{phase} (10 {interval_label} MA)",
                color="red",
                linestyle="--"
            )

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

    # ----------------------------
    # 2) Plot each phase separately
    # ----------------------------
    for phase in phases:
        if not config.PLOT_PHASES.get(phase, False):
            continue
        if phase not in df_phases.columns:
            logging.warning(f"Phase '{phase}' not found in DataFrame columns. Skipping.")
            continue

        phase_series = df_phases[phase].dropna()

        # Start only at the first non-zero date
        non_zero_indices = phase_series[phase_series != 0].index
        if non_zero_indices.empty:
            continue
        first_non_zero_date = non_zero_indices[0]

        # Include data before the first non-zero date for MA calculation
        phase_series_full = df_phases_full[phase].dropna()
        moving_avg = phase_series_full.rolling(window=10, min_periods=1).mean()

        # Trim to start from the first non-zero date for plotting
        phase_series = phase_series.loc[first_non_zero_date:]
        moving_avg = moving_avg.loc[first_non_zero_date:]

        if phase_series.empty:
            continue

        plt.figure(figsize=(10, 6))

        # Plot a line only (no markers)
        plt.plot(phase_series.index, phase_series.values, label=phase)

        # Plot the moving average line
        plt.plot(
            moving_avg.index,
            moving_avg.values,
            label=f"{phase} (10 {interval_label} MA)",
            color="red",
            linestyle="--"
        )

        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("")
        plt.ylabel("")
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    # ----------------------------
    # 3) Correlation Heatmap
    # ----------------------------
    _save_phase_correlation_heatmap(df_phases)

def _save_phase_correlation_heatmap(df_phases):
    """
    Create and save a correlation heatmap for the six phases:
    Bullish, Caution, Distribution, Bearish, Recuperation, Accumulation.
    """

    phases_of_interest = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
    existing_cols = [col for col in phases_of_interest if col in df_phases.columns]

    # If none of these columns exist, skip
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
    Each column is plotted separately (lines only, no markers).
    """
    if not config.PLOT_INDICATORS.get(indicator_name, False):
        return
    if df.empty:
        logging.warning(f"No data to plot for indicator '{indicator_name}'.")
        return

    # Filter for START_DATE and END_DATE
    if config.START_DATE:
        start_date = pd.to_datetime(config.START_DATE)
        df = df[df.index >= start_date]
    if config.END_DATE:
        end_date = pd.to_datetime(config.END_DATE)
        df = df[df.index <= end_date]

    if df.empty:
        logging.warning(f"No data within the specified date range for indicator '{indicator_name}'.")
        return

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    interval_label = _get_interval_label(config.INDICATOR_PLOT_INTERVAL)

    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        plt.figure(figsize=(10, 6))

        # Plot line only
        plt.plot(col_data.index, col_data.values, label=col)

        # If you know this column is a % measure, you can do:
        # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.title(f"{indicator_name} - {col} ({interval_label.upper()} data)")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, f"{indicator_name}_{col}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
