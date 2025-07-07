import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sell.sell_indicator import apply_sell_indicators
from utils.config import MODEL_DIR

def train_sell(df: pd.DataFrame , time_fram_days: int):
    df = apply_sell_indicators(df)

    #sell = 1 in future if price drip then sell
    df["future_close"] = df["Close"].shift(-time_fram_days)
    df["label"] = np.where(df["future_close"] < df["Close"] , 0 ,1)
    df.dropna(inplace=True)

    features = df[["rsi", "macd", "macd_signal", "ema_20", "bollinger_band_upper", "bollinger_band_lower", "obv", "volume_rolling_mean"]]
    target = df["label"]

    X_train , X_test , y_train , y_test = train_test_split(features , target , shuffle=False , test_size=0.2)

    models = {
        "sell_random_forest" : RandomForestClassifier(n_estimators=100 , random_state=42),
        "sell_logistic_regression" : LogisticRegression(max_iter=200),
        "sell_xgboost": XGBClassifier(n_estimators=100 , use_label_encoded = False , eval_matrix = 'logloss')
    }

    result={}
    for name , model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test , model.predict(X_test))
        joblib.load(model , os.path.join(MODEL_DIR ,f"{name}.pkl"))
        result[name] = acc
        print(f"[Sell model:{name}] Accuracy : {acc:.2f}")

    return result




