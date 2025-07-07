import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def predict_price(df : pd.DataFrame , forecast_days:int):
    df = df.copy()
    df["target"] = df["Close"].shift(-forecast_days)
    df.dropna(inplace=True)

    X = np.arange(len(df)).reshape(-1 , 1)
    y = df["target"].values

    lin_model = LinearRegression()
    lin_model.fit(X ,y)

    last_days = X[-1][0]
    future_days = np.arange(last_days + 1 , last_days + forecast_days + 1).reshape(-1 , 1)
    prediction = lin_model.predict(future_days)

    #prepare response
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]

    forecast_result = [
        {"date" : d.strftime("%Y-%m-%d"), "prediction_price" : round(p , 2)}
        for d , p in zip(forecast_dates , prediction)
    ]

    trend = "UPWARD" if prediction[-1] > df["Close"].iloc[-1] else "DOWNWARD"

    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in DataFrame for classification.")


    features = df.drop(columns=["Date", "Open", "High", "Low", "Adj Close", "Close", "Volume", "future_close", "label"] , errors="ignore")
    target = df["label"]

    X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42)

    clf = RandomForestClassifier(n_estimators=100 , random_state=42)

    clf.fit(X_train , y_train)

    #predict
    lastest_data = features.iloc[-1:]
    prediction_label = clf.predict(lastest_data)[0]
    prediction_proba = clf.predict_proba(lastest_data)[0]

    confidence = round(max(prediction_proba), 2)

    return forecast_result , trend , confidence
    