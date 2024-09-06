import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol: str, start_date: str, end_date: str, filename: str):
    """
    Fetch historical stock price data from Yahoo Finance and save it to a CSV file.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL').
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        filename (str): Path to the CSV file where data will be saved.
    """
    # Fetch the data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Save the data to a CSV file
    stock_data.to_csv(filename)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    # Example usage
    fetch_stock_data('AAPL', '2023-01-01', '2024-01-01', 'data/stock_prices.csv')
