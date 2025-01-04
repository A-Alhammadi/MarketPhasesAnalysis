# indicators/adv_decline_volume.py

import pandas as pd

def compute_adv_decline_volume(data_dict):
    # Compute the Advance-Decline Volume Line (Up/Down Volume Line).
    # If a stock closes up vs previous day, add its volume; if down, subtract.
    # Returns DataFrame [Date, AdvDeclVolumeLine]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    volume_line = []
    cumulative = 0
    for date in all_dates:
        up_volume = 0
        down_volume = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    volume = df.iloc[idx]["Volume"]
                    if curr_close > prev_close:
                        up_volume += volume
                    elif curr_close < prev_close:
                        down_volume += volume
        net = up_volume - down_volume
        cumulative += net
        volume_line.append([date, cumulative])
    volume_df = pd.DataFrame(volume_line, columns=["Date", "AdvDeclVolumeLine"])
    volume_df.set_index("Date", inplace=True)
    return volume_df
