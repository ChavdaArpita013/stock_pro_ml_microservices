import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol : str , start_date : str , end_date:str , interval:str):
    df = yf.download(symbol , start=start_date , end=end_date, interval=interval)

    if isinstance(df.columns , pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)
    return df