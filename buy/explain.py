import pandas as pd
from buy.buy_indicators import apply_buy_indicators
from core.explain import get_explaination

def explain_prediction(df: pd.DataFrame , model_name: str = "random_forest"):
    df = apply_buy_indicators(df)
    latest = df[['rsi' , 'macd','sma_50' ,'sma_200' , 'ema_20' , 'ema_50' , 'bb_upper','bb_lower','bb_mavg','obv','volume_change_pct','price_change_pct', 'support','resistance']].iloc[-1]
    
    get_explaination(latest=latest , model_name=model_name)

