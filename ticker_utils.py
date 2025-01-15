# ticker_utils.py

import logging
import requests
from bs4 import BeautifulSoup
import config

def get_sp500_tickers():
    """
    Example: read the list of S&P 500 tickers from Wikipedia.
    If you prefer, you can replace this with a local list or your own logic,
    controlled by config.STOCK_LIST_SOURCE, for instance.
    """
    if config.STOCK_LIST_SOURCE == "WIKI":
        url = config.STOCK_LIST_URL  # <--- replaced hard-coded URL
        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                logging.error("Failed to fetch S&P 500 tickers from Wikipedia.")
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"id": "constituents"})
            if not table:
                logging.error("No S&P 500 table found on Wikipedia page.")
                return []
            rows = table.find_all("tr")[1:]
            tickers = []
            for row in rows:
                cols = row.find_all("td")
                if cols:
                    ticker = cols[0].text.strip()
                    ticker = ticker.replace(".", "-")
                    tickers.append(ticker)
            return tickers
        except Exception as e:
            logging.error(f"Error fetching S&P 500 tickers: {e}")
            return []
    else:
        # If you want to support other data sources in the future,
        # handle them here based on config.STOCK_LIST_SOURCE
        logging.warning(f"STOCK_LIST_SOURCE '{config.STOCK_LIST_SOURCE}' not implemented.")
        return []
