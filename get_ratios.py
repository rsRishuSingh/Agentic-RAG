import requests
import os
from dotenv import load_dotenv

load_dotenv()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def company_overview(symbol: str) -> dict:
    """
    Fetch key company information and financial ratios for a given equity symbol
    from the Alpha Vantage API.

    Args:
        symbol (str): Stock ticker, e.g. "IBM" or "AAPL".

    Returns:
        dict: Parsed JSON response containing company overview data.
    """
    # ensure .env has ALPHAVANTAGE_API_KEY=<your_key>
  

    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Set ALPHAVANTAGE_API_KEY in your environment before calling this tool.")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return data

print(company_overview("AAPL"))
