import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    # Adds multiple technical indicators to the stock dataframe for ML features.
    # Requires columns: Open, High, Low, Close, Volume

    #RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'] , window=14).rsi()
    
    #MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['mscd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    #SMA(50 , 200)
    df['sma_50'] = ta.trend.SMAIndicator(close=df['Close'] , window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(close=df['Close'] , window=200).sma_indicator()

    #EMA(20 , 50)
    df['ema_20'] = ta.trend.EMAIndicator(close=df['Close'] , window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close=df['Close'] , window=50).ema_indicator()

    #BollingerBands
    bb = ta.volatility.BollingerBands(close=df['Close'] , window=20 , window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mavg'] = bb.bollinger_mavg()

    #On-Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'] , volume=df['Volume']).on_balance_volume()

    #volume change %
    df['volume_change_pct'] = df['Volume'].pct_change()

    #price change %
    df['price_change_pct'] = df['Close'].pct_change()

    # Support/Resistance (basic: recent min/max over window)
    df['support'] = df['Close'].rolling(window=20).min()
    df['resistance'] = df['Close'].rolling(window=20).max()

    df = df.dropna()

    return df

