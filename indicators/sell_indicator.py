import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD,EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands

def apply_sell_indicators(df : pd.DataFrame)-> pd.DataFrame:
    df = df.copy()

    if "Close" not in df.columns:
        raise ValueError("'Close' column missing in data.")
    if "Volume" not in df.columns:
        raise ValueError("'Volume' column missing in data.")
    if not isinstance(df["Close"], pd.Series) or df["Close"].ndim != 1:
        print("df['Close'] type:", type(df["Close"]))
        print("df['Close'].ndim:", df["Close"].ndim)
        raise ValueError("'Close' must be a 1D Series")

    #RSI
    rsi = RSIIndicator(close=df["Close"], window=14)

    #MACD
    macd = MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    #EMA
    ema = EMAIndicator(close=df["Close"] , window=20)
    df["ema_20"] = ema.ema_indicator()

    #Bollinger band
    bb = BollingerBands(close=df["Close"] , window=20 , window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()

    #Volume indicator
    obv = OnBalanceVolumeIndicator(close=df["Close"] , volume=df["Volume"])
    df["obv"] = obv.on_balance_volume()

    df = df.dropna()
    return df
    
