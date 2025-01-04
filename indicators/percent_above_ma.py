# indicators/percent_above_ma.py

import pandas as pd

def compute_percent_above_ma(data_dict, ma_window=200):
    # Compute % of tickers above the given moving average window.
    # Returns DataFrame [Date, PercentAboveMA]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    for ticker, df in data_dict.items():
        df.sort_index(inplace=True)
        df[f"SMA_{ma_window}"] = df["Close"].rolling(window=ma_window).mean()

    output = []
    total_tickers = len(data_dict)
    for date in all_dates:
        above_count = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                row = df.loc[date]
                ma_val = row.get(f"SMA_{ma_window}", None)
                close_val = row.get("Close", None)
                if ma_val is not None and close_val is not None and close_val > ma_val:
                    above_count += 1
        pct = (above_count / total_tickers) * 100 if total_tickers else 0
        output.append([date, pct])
    pct_df = pd.DataFrame(output, columns=["Date", "PercentAboveMA"])
    pct_df.set_index("Date", inplace=True)
    return pct_df
