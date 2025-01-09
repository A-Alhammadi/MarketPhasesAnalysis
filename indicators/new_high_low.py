# indicators/new_high_low.py

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def compute_new_high_low(data_dict, lookback=252):
    """
    Compute how many tickers made a new ~52-week high or low each day.
    Returns DataFrame [Date, NewHighCount, NewLowCount, NHNL_Diff, NHNL_Ratio].
    """
    all_dates = sorted(
        set(
            date
            for df in data_dict.values()
            for date in df.index
        )
    )

    rolling_info = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            logging.warning(f"Ticker {ticker} is missing 'Close' column. Skipping.")
            continue

        # Ensure the DataFrame index is unique
        df = df[~df.index.duplicated(keep="first")]
        df_sorted = df.sort_index()

        # Calculate rolling max and min
        df_sorted["RollingMax"] = df_sorted["Close"].rolling(window=lookback, min_periods=1).max()
        df_sorted["RollingMin"] = df_sorted["Close"].rolling(window=lookback, min_periods=1).min()
        rolling_info[ticker] = df_sorted

    output = []
    for date in all_dates:
        new_high_count = 0
        new_low_count = 0

        for ticker, df_ in rolling_info.items():
            if date in df_.index:
                row = df_.loc[date]

                # Handle multiple rows for the same date
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                # Extract scalar values and skip if NaN
                close = row.get("Close")
                rolling_max = row.get("RollingMax")
                rolling_min = row.get("RollingMin")

                if pd.isna(close) or pd.isna(rolling_max) or pd.isna(rolling_min):
                    continue

                # Check for new highs/lows
                if close == rolling_max:
                    new_high_count += 1
                if close == rolling_min:
                    new_low_count += 1

        nhnl_diff = new_high_count - new_low_count
        nhnl_ratio = (new_high_count / new_low_count) if new_low_count else None
        output.append([date, new_high_count, new_low_count, nhnl_diff, nhnl_ratio])

    nhnl_df = pd.DataFrame(
        output,
        columns=["Date", "NewHighCount", "NewLowCount", "NHNL_Diff", "NHNL_Ratio"]
    )
    nhnl_df.set_index("Date", inplace=True)
    return nhnl_df


