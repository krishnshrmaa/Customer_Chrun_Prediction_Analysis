from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor
try:
    model = joblib.load('models/best_model_logistic_regression.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    preprocessor = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict")
async def predict(data: dict):
    """
    Predict customer churn
    
    Example input:
    {
        "tenure": 2,
        "MonthlyCharges": 65.1,
        "TotalCharges": 130.2,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    """
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input dict to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        return {
            "churn_prediction": int(prediction),
            "churn_label": "Yes" if prediction == 1 else "No",
            "probability_no_churn": float(probability[0]),
            "probability_churn": float(probability[1])
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
