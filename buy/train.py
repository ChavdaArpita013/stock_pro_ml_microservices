import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from buy.buy_indicators import apply_buy_indicators
from utils.config import MODEL_DIR

def train_buy_model(df: pd.DataFrame):
    df = apply_buy_indicators(df)
    df["target"] = (df["Close"].shift(-5) > df["Close"].astype(int))
    df = df.dropna(inplace=True)

    X = df[['rsi' , 'macd','sma_50' ,'sma_200' , 'ema_20' , 'ema_50' , 'bb_upper','bb_lower','bb_mavg','obv','volume_change_pct','price_change_pct', 'support','resistance']]
    y = df["target"]

    models = {
        "random_forest" : RandomForestClassifier(n_estimators=100),
        "logistic_regression" : LogisticRegression(max_iter=200),
        "xgboost" : XGBClassifier(n_estimators=100 , use_label_encoded = False , eval_matrix = 'logloss')
    }

    result = {}

    for name, model in models.items():
        model.fit(X , y)
        acc = accuracy_score(y , model.predict(X))
        joblib.dump(model , os.path.join(MODEL_DIR , f"{name}BUY.pkl"))
        result[name] = acc
        print(f"[Buy Model: {name} Accuracy: {acc:2f}]")
    
    return result
        
