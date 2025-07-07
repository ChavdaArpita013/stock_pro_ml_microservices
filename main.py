from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from core.data_loader import fetch_stock_data
from model.forecast_model import predict_price
from indicators.forecast_indicator import apply_forecast_indicator
from buy.predict import predict_buy
from sell.predict import predict_sell
from sell.train import train_sell
from sell.explain import explain_sell_prediction

app = FastAPI()

class BuyRequest(BaseModel):
    symbol :str
    start : str
    end : str
    interval: str


@app.post("/buy-prediction")
def predict_stock(req: BuyRequest):

    try:
        df = fetch_stock_data(
            symbol=req.symbol,
            start_date="2024-01-01",
            end_date=req.currDate,
            interval=req.interval
        )

        if df.empty or len < 60:
            raise HTTPException(status_code=500 , detail="No enough data to predict")

        if "Close" not in df.columns:
            raise ValueError("Missing required column Close in request")
        
        final_decision , model_votes = predict_buy(df)
        
        return{
            "symbol" : req.symbol,
            "final_decision": "BUY" if final_decision == 1 else "HOLD",
            "model_votes" : model_votes,
            "confidence_percentage" : f"{model_votes.count(1)/len(model_votes) * 100 : .2f}%",
            "status": "success" 
        }

    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))

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
    try:
        df = fetch_stock_data(
            symbol=req.symbol,
            start_date="2024-01-01",
            end_date=req.currDate,
            interval=req.interval
        )

        if df.empty or len < 60:
            raise HTTPException(status_code=500 , detail="No enough data to predict")
        
        train_sell(df , req.timeFrameDays)
        prediction = predict_sell(df)

        latest = df.iloc[-1]
        reasons = explain_sell_prediction(latest)

        return{
            "symbol": req.symbol,
            "final_decision" : prediction["final_decision"],
            "confidence": prediction["confidence"],
            "model_votes" : prediction["model_votes"],
            "reasons": reasons,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))

class ForecastRequest(BaseModel):
    symbol: str
    start: str
    interval : str
    forecastDays : int

def get_stock_forecast(req : ForecastRequest):
    df = fetch_stock_data(req.symbol , req.start , req.interval)
    forecast, trend, confidence = predict_price(df, req.forecastDays)
    indicators = apply_forecast_indicator(df)

    return{
        "forecast" : forecast,
        "trend": trend,
        "confidence": confidence,
        "technical_indicators": indicators
    }
