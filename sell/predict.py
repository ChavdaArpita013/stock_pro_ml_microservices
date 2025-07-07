import os
from utils.config import MODEL_DIR
from core.ensemble import majority_vote
import pandas as pd
import joblib
from sell.sell_indicator import apply_sell_indicators

def predict_sell(df: pd.DataFrame):
    df = apply_sell_indicators(df)
    latest = df.iloc[-1]
    
    features = df[["rsi", "macd", "macd_signal", "ema_20", "bollinger_band_upper", "bollinger_band_lower", "obv", "volume_rolling_mean"]]

    models = []
    votes = []
    for name in ["sell_random_forest", "sell_logistic_regression" , "sell_xgboost"]:
        path = os.path.join(MODEL_DIR , f"{name}.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            prediction = model.predict(features)[0]
            votes.append(prediction)
            models.append(model)

    if not models :
        raise ValueError("No sell model found")
    
    decision = majority_vote(votes)
    confidence = f"{votes.count(1)/len(votes) * 100: .2f}"

    return {
        "final_decision": "SELL" if decision == 1 else 0,
        "model_votes" : votes,
        "confidence": confidence
    }
