# analysis/indicators/price_indicators.py

import logging
import psycopg2.extras
import pandas as pd
import numpy as np
import config

def compute_and_store_price_ma_deviation(data_dict: dict, conn):
    """
    Calculate how far Close is in % from its 50-day & 200-day MAs, 
    store in price_ma_deviation.
    """
    if not data_dict:
        return
    
    cur = conn.cursor()
    records = []
    for ticker, df in data_dict.items():
        if df.empty or "Close" not in df.columns:
            continue
        
        df.sort_index(inplace=True)
        df["Close"] = df["Close"].astype(float)
        
        df["ma_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        df["ma_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
        
        df["dev_50"] = ((df["Close"] - df["ma_50"]) / df["ma_50"]) * 100
        df["dev_200"] = ((df["Close"] - df["ma_200"]) / df["ma_200"]) * 100
        
        for dt_, row in df.iterrows():
            data_date = dt_.date()
            dev50 = row["dev_50"]
            dev200 = row["dev_200"]
            
            dev50 = float(dev50) if pd.notna(dev50) else None
            dev200 = float(dev200) if pd.notna(dev200) else None
            
            # Skip earliest NaNs if desired
            if dev50 is None or dev200 is None:
                continue
            
            records.append((ticker, data_date, dev50, dev200))
    
    insert_query = """
        INSERT INTO price_ma_deviation
          (ticker, data_date, dev_50, dev_200)
        VALUES %s
        ON CONFLICT (ticker, data_date) DO UPDATE
          SET dev_50 = EXCLUDED.dev_50,
              dev_200 = EXCLUDED.dev_200
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()
