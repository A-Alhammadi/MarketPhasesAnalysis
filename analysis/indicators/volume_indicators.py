# volume_indicators.py

import logging
import os
import pandas as pd
import psycopg2.extras
import config
import numpy as np


def compute_and_store_volume_mas(data_dict: dict, conn):
    """
    For each ticker, compute the 10-day and 20-day volume MAs, store in volume_ma_data table.
    Skips any rows with missing/NaN/zero volume.
    """
    if not data_dict:
        return

    cur = conn.cursor()
    records = []

    for ticker, df in data_dict.items():
        # Must have data and a Volume column
        if df.empty or "Volume" not in df.columns:
            continue

        # Drop rows where Volume is NaN or zero
        df = df.dropna(subset=["Volume"])
        df = df[df["Volume"] != 0]
        if df.empty:
            continue

        df.sort_index(inplace=True)
        df["Volume"] = df["Volume"].astype(float)

        df["vol_ma_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
        df["vol_ma_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

        for dt_, row in df.iterrows():
            trade_date = dt_.date()
            vol_ma10 = row["vol_ma_10"]
            vol_ma20 = row["vol_ma_20"]

            # If either MA is NaN, skip
            if pd.isna(vol_ma10) or pd.isna(vol_ma20):
                continue

            records.append((ticker, trade_date, float(vol_ma10), float(vol_ma20)))

    if records:
        insert_query = """
            INSERT INTO volume_ma_data
              (ticker, trade_date, vol_ma_10, vol_ma_20)
            VALUES %s
            ON CONFLICT (ticker, trade_date) DO UPDATE
              SET vol_ma_10 = EXCLUDED.vol_ma_10,
                  vol_ma_20 = EXCLUDED.vol_ma_20
        """
        psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
        conn.commit()
    else:
        logging.info("No valid records to insert into volume_ma_data.")

    cur.close()


def compute_and_store_volume_ma_deviation(data_dict: dict, conn):
    """
    Calculate how far Volume is from its 20-day & 63-day MAs, store in volume_ma_deviation.
    Skips rows with missing/NaN/zero volume or missing MAs.
    """
    if not data_dict:
        return

    cur = conn.cursor()
    records = []

    for ticker, df in data_dict.items():
        if df.empty or "Volume" not in df.columns:
            continue

        # Drop rows where Volume is NaN or zero
        df = df.dropna(subset=["Volume"])
        df = df[df["Volume"] != 0]
        if df.empty:
            continue

        df.sort_index(inplace=True)
        df["Volume"] = df["Volume"].astype(float)

        df["vol_ma_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["vol_ma_63"] = df["Volume"].rolling(window=63, min_periods=1).mean()

        df["dev_20"] = ((df["Volume"] - df["vol_ma_20"]) / df["vol_ma_20"]) * 100
        df["dev_63"] = ((df["Volume"] - df["vol_ma_63"]) / df["vol_ma_63"]) * 100

        for dt_, row in df.iterrows():
            data_date = dt_.date()
            d20 = row["dev_20"]
            d63 = row["dev_63"]

            # If dev_20 or dev_63 is NaN, skip
            if pd.isna(d20) or pd.isna(d63):
                continue

            records.append((ticker, data_date, float(d20), float(d63)))

    if records:
        insert_query = """
            INSERT INTO volume_ma_deviation
              (ticker, data_date, dev_20, dev_63)
            VALUES %s
            ON CONFLICT (ticker, data_date) DO UPDATE
              SET dev_20 = EXCLUDED.dev_20,
                  dev_63 = EXCLUDED.dev_63
        """
        psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
        conn.commit()
    else:
        logging.info("No valid records to insert into volume_ma_deviation.")

    cur.close()


def export_extreme_volumes(conn):
    """
    Exports stocks with extreme volume deviations.
    Uses config.EXTREME_VOLUME_Z_THRESHOLD for the Z-score threshold.
    """
    cur = conn.cursor()

    # 1) Find the last date in volume_ma_deviation
    cur.execute("SELECT MAX(data_date) FROM volume_ma_deviation;")
    row = cur.fetchone()
    if not row or not row[0]:
        logging.warning("No data in volume_ma_deviation table.")
        cur.close()
        return
    last_date = row[0]  # date object

    # 2) Query the data for that date
    query = """
        SELECT ticker, dev_20, dev_63
        FROM volume_ma_deviation
        WHERE data_date = %s
          AND dev_20 IS NOT NULL
          AND dev_63 IS NOT NULL
    """
    cur.execute(query, (last_date,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logging.info(f"No volume deviation data on {last_date}.")
        return

    # 3) Create a DataFrame
    df = pd.DataFrame(rows, columns=["Ticker", "Dev_20", "Dev_63"])
    df["Dev_20"] = df["Dev_20"].astype(float)
    df["Dev_63"] = df["Dev_63"].astype(float)

    # Compute z-scores
    df["Dev_20_Z"] = (df["Dev_20"] - df["Dev_20"].mean()) / df["Dev_20"].std()
    df["Dev_63_Z"] = (df["Dev_63"] - df["Dev_63"].mean()) / df["Dev_63"].std()

    # Use threshold from config
    z_thresh = config.EXTREME_VOLUME_Z_THRESHOLD
    extremes = df[
        (df["Dev_20_Z"].abs() >= z_thresh) | (df["Dev_63_Z"].abs() >= z_thresh)
    ]

    # 4) Write results to a file
    if not extremes.empty:
        file_name = os.path.join(config.RESULTS_DIR, f"extreme_volume_stocks_{last_date}.txt")
        with open(file_name, "w") as f:
            f.write(f"Extreme Volume Stocks for {last_date} (Z-Threshold: {z_thresh}):\n\n")
            for _, row in extremes.iterrows():
                f.write(
                    f"{row['Ticker']}: Dev_20_Z = {row['Dev_20_Z']:.2f}, "
                    f"Dev_63_Z = {row['Dev_63_Z']:.2f}\n"
                )
        logging.info(f"Extreme volume file created: {file_name}")
    else:
        logging.info(f"No stocks exceeded z-threshold {z_thresh} on {last_date}.")