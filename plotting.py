import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import config

matplotlib.use("Agg")

def save_phase_plots(phases, df_phases):
    """
    Generate and save plots for all phases in df_phases (resampled or daily).
    Ensures consistent y-axis (0% to 100%) and overlays a red 10-day moving average.
    """
    if df_phases.empty:
        logging.warning("No phase data to plot.")
        return
    
    # Filter for START_DATE and END_DATE
    if config.START_DATE:
        start_date = pd.to_datetime(config.START_DATE)
        df_phases = df_phases[df_phases.index >= start_date]
    if config.END_DATE:
        end_date = pd.to_datetime(config.END_DATE)
        df_phases = df_phases[df_phases.index <= end_date]

    if df_phases.empty:
        logging.warning("No phase data within the specified date range to plot.")
        return

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    # Plot all phases together
    if config.PLOT_PHASES.get("AllPhases", False):
        plt.figure(figsize=(12, 8))
        for phase in phases:
            if phase in df_phases.columns:
                plt.plot(df_phases.index, df_phases[phase], label=phase)
                # 10-day moving average in red
                moving_avg = df_phases[phase].rolling(window=10).mean()
                plt.plot(df_phases.index, moving_avg, label=f"{phase} (10d MA)", color="red", linestyle="--")
        plt.title("All Phases Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.ylim(0, 100)  # Set consistent y-axis for percentage
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, "all_phases.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    # Plot each phase individually
    for phase in phases:
        if not config.PLOT_PHASES.get(phase, False):
            continue
        if phase not in df_phases.columns:
            logging.warning(f"Phase '{phase}' not found in DataFrame columns. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(df_phases.index, df_phases[phase], label=phase)
        # 10-day moving average
        moving_avg = df_phases[phase].rolling(window=10).mean()
        plt.plot(df_phases.index, moving_avg, label=f"{phase} (10d MA)", color="red", linestyle="--")

        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("")
        plt.ylabel("")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
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

    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        # Plot from the first non-default value
        first_val = col_data.iloc[0]
        changed_mask = col_data.ne(first_val)
        if changed_mask.any():
            first_change_idx = changed_mask.idxmax()
            col_data = col_data.loc[first_change_idx:]

        plt.figure(figsize=(10, 6))
        plt.plot(col_data.index, col_data.values, label=col)
        plt.title(f"{indicator_name} - {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, f"{indicator_name}_{col}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
