import pandas as pd
import ta
import ta.momentum
import ta.trend

def apply_forecast_indicator(df : pd.DataFrame)-> dict:
    df = df.copy()
    indicators = {}

    #EMA 50 & 200
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"] , window=50)
    df["EMA_200"] = ta.trend.ema_indicator(df["Close"] , window=200)

    #Golden cross
    if df["EMA_50"].iloc[-1] > df["EMA_200"].iloc[-1]:
        indicators["ema_cross"] = "Golden Cross (Bullish)"
    else:
        indicators["ema_cross"] = "Death Cross (bearish)"

    #ADX
    adx = ta.trend.adx(df["High"] , df["Low"] , df["Close"] , window=14)
    indicators["adx"] = round(adx.iloc[-1] , 2)
    indicators["adx_trend"] = "Strong Trend" if adx.iloc[-1] > 25 else "Weak/No Trend"

    #RSI
    rsi = ta.momentum.RSIIndicator(df["Close"] , window=14).rsi()
    indicators['RSI'] = round(rsi.iloc[-1] , 2)
    if df['RSI'].iloc[-1] > 70:
        indicators["rsi_status"] = "Overbought"
    elif df['RSI'].iloc[-1] < 30:
        indicators["rsi_status"] = "Oversold"
    else:
        indicators["rsi_status"] = "Neuteral"

    #Ichimocku Cloud(tenkan-sen, kijun-sen — simplified)
    df["tenkan_sen"] = (df["High"].rolling(9).max() + df["Low"].rolling(9).min())/2
    df["kijun_sen"] = (df["High"].rolling(26).max() + df["Low"].rolling(26).min())/2

    if df["tenkan_sen"].iloc[-1] > df["kijun_sen"].iloc[-1]:
        indicators["ichimoku_trend"] = "Bullish bias"
    else:
        indicators["ichimoku_trend"] = "Bearish bias"

    return indicators
