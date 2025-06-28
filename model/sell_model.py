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

    return "SELL" if prediction == 1 else "HOLD"

    