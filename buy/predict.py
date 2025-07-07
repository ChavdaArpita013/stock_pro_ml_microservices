import os 
import joblib
from buy.buy_indicators import apply_buy_indicators
from utils.config import MODEL_DIR
from core.ensemble import majority_vote

def predict_buy(df):
    df = apply_buy_indicators(df)
    X = df[['rsi' , 'macd','sma_50' ,'sma_200' , 'ema_20' , 'ema_50' , 'bb_upper','bb_lower','bb_mavg','obv','volume_change_pct','price_change_pct', 'support','resistance']]

    models = []
    for name in ["random_forest" , "logistic_regression" , "xgboost"]:
        path = os.path.join(MODEL_DIR , f"{name}.pkl")
        models.append(joblib.load(path))

    prediction = [model.predict(X)[0] for model in models]
    final_prediction = majority_vote(prediction)
    return final_prediction, prediction
