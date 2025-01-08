# indicators/trin.py

import logging
import pandas as pd
from typing import Dict

def compute_trin(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes the TRIN (Arms Index):
      TRIN = (Advancers / Decliners) / (AdvVolume / DeclVolume)
    If there are zeros, results may be NaN or infinite.
    """
    # Collect all dates
    all_dates = sorted({date for df in data_dict.values() for date in df.index})

    data_output = []
    for date in all_dates:
        advancers = 0
        decliners = 0
        adv_volume = 0
        decl_volume = 0

        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    volume = df.iloc[idx]["Volume"]
                    if any(pd.isna(x) for x in [prev_close, curr_close, volume]):
                        continue

                    if curr_close > prev_close:
                        advancers += 1
                        adv_volume += volume
                    elif curr_close < prev_close:
                        decliners += 1
                        decl_volume += volume

        # Calculate TRIN
        if decliners == 0 or decl_volume == 0:
            trin_val = float('nan')
        else:
            numerator = (advancers / decliners) if decliners else float('inf')
            denominator = (adv_volume / decl_volume) if decl_volume else float('inf')
            trin_val = numerator / denominator

        data_output.append([date, trin_val])

    trin_df = pd.DataFrame(data_output, columns=["Date", "TRIN"])
    trin_df.set_index("Date", inplace=True)
    return trin_df
