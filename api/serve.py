from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving.langgraph_support import build_support_graph

app = FastAPI(
    title="E-Commerce AI Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Load model
model = None

class ForecastRequest(BaseModel):
    days: int = 30

class ForecastResponse(BaseModel):
    predictions: List[float]
    dates: List[str]
    total_forecast: float
    avg_daily: float

class SupportRequest(BaseModel):
    message: str

class SupportResponse(BaseModel):
    intent: str
    response: str
    escalate: bool
    reasoning: str

@app.on_event("startup")
async def load_model():
    global model
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("ecommerce_demand")
        if experiment:
            runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rmse ASC"])
            if runs:
                model_uri = f"runs:/{runs[0].info.run_id}/random_forest_model"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"Model loaded from {model_uri}")
            else:
                print("No model found")
        else:
            print("No experiment found")
    except Exception as e:
        print(f"Error loading model: {e}")

# Initialize LangGraph
support_graph = build_support_graph()

@app.get("/")
async def root():
    return {
        "message": "E-Commerce AI Platform",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "support": "/support (POST)",
            "forecast": "/forecast/daily (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/forecast/daily", response_model=ForecastResponse)
async def forecast_daily(request: ForecastRequest):
    start_date = datetime.now()
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.days)]
    
    base = 300
    trend = np.linspace(0, 50, request.days)
    seasonality = 50 * np.sin(np.linspace(0, 4*np.pi, request.days))
    predictions = (base + trend + seasonality).tolist()
    
    return ForecastResponse(
        predictions=predictions,
        dates=dates,
        total_forecast=sum(predictions),
        avg_daily=sum(predictions) / len(predictions)
    )

@app.post("/support", response_model=SupportResponse)
async def customer_support(request: SupportRequest):
    result = support_graph.invoke({
        "customer_message": request.message,
        "intent": "",
        "sentiment": "",
        "response": "",
        "escalate": False,
        "reasoning": "",
        "messages": []
    })
    
    return SupportResponse(
        intent=result["intent"],
        response=result["response"],
        escalate=result.get("escalate", False),
        reasoning=result.get("reasoning", "")
    )

@app.get("/info")
async def info():
    return {
        "name": "E-Commerce AI Platform",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/forecast/daily", "/support", "/info", "/docs", "/redoc"],
        "model_ready": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)