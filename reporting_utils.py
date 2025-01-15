# reporting_utils.py

import os
import logging
import pandas as pd
import config

def save_latest_breadth_values(
    df_phases_daily: pd.DataFrame,
    all_indicators: dict,
    output_file="breadth_values.txt"
):
    """
    Writes the most recent phases breakdown and indicator
    values (including z-score & percentile) to a text file.
    """
    with open(output_file, "w") as f:
        f.write("=== Latest Phase Breakdown ===\n")
        if not df_phases_daily.empty:
            latest_phases = df_phases_daily.iloc[-1]
            date_str = df_phases_daily.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {date_str}\n")
            for phase_col in latest_phases.index:
                f.write(f"{phase_col}: {latest_phases[phase_col]:.2f}%\n")
        else:
            f.write("No phase data available.\n")

        for indicator_name, df_data in all_indicators.items():
            if df_data.empty:
                continue
            f.write("\n")
            f.write(f"=== Latest Values for {indicator_name} ===\n")
            last_row = df_data.iloc[-1]
            date_str = df_data.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {date_str}\n")

            for col in df_data.columns:
                col_series = df_data[col].dropna()
                if col_series.empty:
                    continue
                latest_val = last_row[col]
                col_mean = col_series.mean()
                col_std = col_series.std()
                z_score = (latest_val - col_mean) / col_std if col_std != 0 else 0.0
                percentile = (col_series <= latest_val).mean() * 100
                f.write(
                    f"{col} = {latest_val:.4f}, "
                    f"Z-score = {z_score:.4f}, "
                    f"Percentile = {percentile:.2f}%\n"
                )


def save_phases_breakdown(
    df_phases_daily: pd.DataFrame,
    df_phases_resampled: pd.DataFrame,
    output_file=os.path.join(config.RESULTS_DIR, "phases_breakdown.txt")
):
    """
    Save a detailed breakdown of phases to a text file.
    Includes both daily and resampled data.
    """
    with open(output_file, "w") as f:
        f.write("=== Phases Breakdown ===\n\n")

        # Daily Phases Breakdown
        f.write(">> Daily Phases Breakdown:\n")
        if not df_phases_daily.empty:
            latest_daily = df_phases_daily.iloc[-1]
            latest_date = df_phases_daily.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {latest_date}\n")
            for phase, value in latest_daily.items():
                f.write(f"{phase}: {value:.2f}%\n")
        else:
            f.write("No daily phase data available.\n")

        f.write("\n")

        # Resampled Phases Breakdown
        f.write(">> Resampled Phases Breakdown:\n")
        if not df_phases_resampled.empty:
            for idx, row in df_phases_resampled.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                f.write(f"Date: {date_str}\n")
                for phase, value in row.items():
                    f.write(f"  {phase}: {value:.2f}%\n")
                f.write("\n")
        else:
            f.write("No resampled phase data available.\n")

    logging.info(f"Phases breakdown saved to {output_file}")
