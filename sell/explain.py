import pandas as pd
from sell.sell_indicator import apply_sell_indicators
from core.explain import get_explaination

def explain_sell_prediction(df: pd.DataFrame, model_name: str = "sell_rf"):
    df = apply_sell_indicators(df)
    features = ["rsi", "macd", "macd_signal", "ema_20", "bollinger_band_upper", "bollinger_band_lower", "obv", "volume_rolling_mean"]
    latest = df[features].iloc[-1:]

    get_explaination(latest=latest , model_name=model_name)