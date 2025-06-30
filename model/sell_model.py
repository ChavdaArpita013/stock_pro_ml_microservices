import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from indicators.sell_indicator import apply_sell_indicators
from utils.data_loader import fetch_stock_data

def train_and_predict_sell(symbol: str, buy_price:float , buy_date: str , curr_price : float , curr_date:str , interval:str , time_frame_days: int):
    #Load historic data
    df = fetch_stock_data(symbol=symbol , start_date="2024-01-01" , end_date=curr_date , interval=interval)
    if df.empty or len(df) < 60:
        return "no enough data train"
    
    #apply indicator
    df = apply_sell_indicators(df)

    #create sell label (SELL or HOLD)
    df["future_close"] = df["Close"].shift(-time_frame_days)
    df["label"] = np.where(df["future_close"] > df["Close"] , 0, 1)

    features = df.drop(columns=["Date", "Open", "High", "Low", "Adj Close", "Close", "Volume", "future_close", "label"] , errors="ignore")
    target = df["label"]

    X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42)

    model = RandomForestClassifier(n_estimators=100 , random_state=42)
    model.fit(X_train , y_train)

    #predict for latest data
    latest_data = features.iloc[-1:]
    prediction = model.predict(latest_data)[0]
    probabilities = model.predict_proba(latest_data)[0]
    confidence = round(probabilities[prediction] , 2)

    #Generate reason for decision made by model
    reasons = [] 
    last_row = df.iloc[-1]

    # MACD bearish crossover
    if last_row.get("macd") is not None and last_row.get("macd_signal") is not None:
        if last_row["macd"] < last_row["macd_signal"]:
            reasons.append("MACD bearish crossover")

    # RSI overbought
    if last_row.get("rsi") is not None and last_row["rsi"] > 70:
        reasons.append("RSI indicates overbought")

    # Bollinger Band upper breakout
    if last_row.get("bollinger_band_upper") is not None and last_row.get("Close") is not None:
        if last_row["Close"] > last_row["bollinger_band_upper"]:
            reasons.append("Price crossed upper Bollinger Band")

    # Price below EMA (20)
    if last_row.get("ema_20") is not None and last_row.get("Close") is not None:
        if last_row["Close"] < last_row["ema_20"]:
            reasons.append("Price fell below EMA-20")

    # High volume drop
    if last_row.get("volume_rolling_mean") is not None and last_row.get("Volume") is not None:
        if last_row["Volume"] > 1.5 * last_row["volume_rolling_mean"]:
            reasons.append("Unusually high volume with price drop")

    # Pick top 2 only
    if not reasons:
        reasons.append("Based on historical pattern")

    return {
        "decision": "SELL" if prediction == 1 else "HOLD",
        "confidence": confidence,
        "reason": reasons[:2]  # limit to top 2
    }

    