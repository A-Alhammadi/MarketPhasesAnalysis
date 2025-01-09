def classify_phases(data_dict, ma_short=50, ma_long=200):
    """
    Classify each ticker's daily state into phases, then compute a percentage
    distribution of each phase across all tickers on each date.
    """
    import logging
    import pandas as pd

    for ticker, df in data_dict.items():
        if df.empty:
            logging.warning(f"Ticker {ticker} has empty data. Skipping.")
            continue

        # Ensure the data is sorted
        df.sort_index(inplace=True)

        # Check for required 'Close' column
        if "Close" not in df.columns:
            logging.warning(f"Ticker {ticker} is missing 'Close' column. Skipping.")
            continue

        # Calculate rolling means for short & long windows
        df[f"SMA_{ma_short}"] = df["Close"].rolling(window=ma_short, min_periods=1).mean()
        df[f"SMA_{ma_long}"] = df["Close"].rolling(window=ma_long, min_periods=1).mean()

        # Remove duplicate indices (keep the first occurrence for each date)
        df = df[~df.index.duplicated(keep="first")]

    # Gather all dates present in any DataFrame
    all_dates = sorted(
        set(
            date
            for df in data_dict.values()
            for date in df.index
        )
    )

    # Initialize classification counts
    classification_counts = {
        date: {
            "Bullish": 0,
            "Caution": 0,
            "Distribution": 0,
            "Bearish": 0,
            "Recuperation": 0,
            "Accumulation": 0
        }
        for date in all_dates
    }

    # Tally each ticker's phase on each date
    for ticker, df in data_dict.items():
        if df.empty:
            continue

        for date in df.index:
            try:
                # Retrieve scalar values for the current date
                sma_short_val = df.loc[date, f"SMA_{ma_short}"]
                sma_long_val = df.loc[date, f"SMA_{ma_long}"]
                price = df.loc[date, "Close"]

                # Ensure they are scalar values (convert Series to scalar if necessary)
                if isinstance(sma_short_val, pd.Series):
                    sma_short_val = sma_short_val.iloc[0]
                if isinstance(sma_long_val, pd.Series):
                    sma_long_val = sma_long_val.iloc[0]
                if isinstance(price, pd.Series):
                    price = price.iloc[0]

                # Skip rows with NaN values
                if pd.isna(sma_short_val) or pd.isna(sma_long_val) or pd.isna(price):
                    continue

                # Basic logic for phases
                if (sma_short_val > sma_long_val) and (price > sma_short_val) and (price > sma_long_val):
                    classification_counts[date]["Bullish"] += 1
                elif (sma_short_val > sma_long_val) and (price < sma_short_val):
                    if price < sma_long_val:
                        classification_counts[date]["Distribution"] += 1
                    else:
                        classification_counts[date]["Caution"] += 1
                elif (sma_short_val < sma_long_val) and (price < sma_short_val) and (price < sma_long_val):
                    classification_counts[date]["Bearish"] += 1
                elif (sma_short_val < sma_long_val) and (price > sma_short_val):
                    if price > sma_long_val:
                        classification_counts[date]["Accumulation"] += 1
                    else:
                        classification_counts[date]["Recuperation"] += 1

            except KeyError:
                continue  # Missing data for this date

    # Convert tally counts to percentages
    total_tickers = len(data_dict)
    rows = []
    for date in sorted(all_dates):
        row = [date]
        for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
            count = classification_counts[date][phase]
            pct = (count / total_tickers) * 100 if total_tickers > 0 else 0
            row.append(pct)
        rows.append(row)

    # Create the final DataFrame
    col_names = [
        "Date",
        "Bullish", "Caution", "Distribution",
        "Bearish", "Recuperation", "Accumulation"
    ]
    df_phases = pd.DataFrame(rows, columns=col_names)
    df_phases.set_index("Date", inplace=True)

    return df_phases
