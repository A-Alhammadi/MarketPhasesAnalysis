# indicators/adv_decline.py

import logging
import pandas as pd

def compute_adv_decline(data_dict):
    """
    Compute the Advance-Decline Line for all tickers combined.
    """
    all_dates = sorted(
        list(
            set(
                date for df in data_dict.values() for date in df.index
            )
        )
    )
    ad_line = []
    cumulative = 0

    for date in all_dates:
        advancers = 0
        decliners = 0
        for ticker, df in data_dict.items():
            if "Close" not in df.columns:
                logging.warning(f"Ticker {ticker} is missing 'Close' column on {date}. Skipping.")
                continue

            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    if pd.isna(prev_close) or pd.isna(curr_close):
                        continue
                    if curr_close > prev_close:
                        advancers += 1
                    elif curr_close < prev_close:
                        decliners += 1

        net = advancers - decliners
        cumulative += net
        ad_line.append([date, cumulative])

    ad_df = pd.DataFrame(ad_line, columns=["Date", "AdvanceDeclineLine"])
    ad_df.set_index("Date", inplace=True)
    return ad_df