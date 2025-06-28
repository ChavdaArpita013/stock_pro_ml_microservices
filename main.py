from fastapi import FastAPI
from pydantic import BaseModel
from indicators.buy_indicators import apply_buy_indicators
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model.sell_model import train_and_predict_sell

app = FastAPI()

class BuyRequest(BaseModel):
    symbol :str
    start : str
    end : str


@app.post("/buy-prediction")
def predict_stock(data: BuyRequest):

    df = yf.download(data.symbol , start=data.start , end=data.end)

    if(df.empty):
        return {"error: No Stock data avaible"}
    df = apply_buy_indicators(df)

    features = df[[
        'Close', 'Volume', 'rsi', 'macd', 'macd_signal',
        'sma_50', 'sma_200', 'ema_20', 'ema_50',
        'bb_upper', 'bb_lower', 'bb_mavg', 'obv',
        'volume_change_pct', 'price_change_pct',
        'support', 'resistance'
    ]].values

    window_size = 5
    if len(features) < window_size:
        return {"No Enough data avaible to make prediction"}
    
    X = []
    y =[]

    for i in range(len(features) - window_size - 1):
        window = features[i:i+window_size]
        curr_close = features[i + window_size - 1][0]
        next_close = features[i + window_size ][0]
        label = 1 if next_close > curr_close else 0
        X.append(window)
        y.append(label)

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=100 , random_state=42)
    model.fit(X,y)
    
    latest_window = features[-window_size:]
    input_data = latest_window.reshape(1, -1)

    prediction = model.predict(input_data)[0]
    advice = "BUY" if prediction == 1 else "WAIT"

    return{
        "symbol" : data.symbol,
        "date": str(df.index[-1].date()),
        "advice":advice,
        "prediction": int(prediction)
    }

class SellReqest(BaseModel):
    symbol: str
    buyPrice: float
    buyDate: str
    currPrice: float
    currDate: str
    interval: str
    timeFrameDays: int

@app.post("/sell-prediction")
def predict_sell(req: SellReqest):
    result = train_and_predict_sell(
        symbol=req.symbol,
        buy_price=req.buyPrice,
        buy_date=req.buyDate,
        curr_price=req.currPrice,
        curr_date=req.currDate,
        interval=req.interval,
        time_frame_days=req.timeFrameDays
    )
    return {"action" : result}