# ticker_utils.py

import logging

def get_sp500_tickers():
    """
    Example: read the list of S&P 500 tickers from Wikipedia.
    If you prefer, you can replace this with a hard-coded list or your own logic.
    """
    import requests
    from bs4 import BeautifulSoup
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            logging.error("Failed to fetch S&P 500 tickers from Wikipedia.")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if not table:
            logging.error("No S&P 500 table found on Wikipedia.")
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
