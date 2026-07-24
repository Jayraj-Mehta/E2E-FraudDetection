import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from app.config import settings
from app.schemas import CustomerInput, PredictionOutput

# ------------------------------------------------------------------
# 1. THE ML PREDICTOR WRAPPER
# ------------------------------------------------------------------
class SegmentationPredictor:
    """Manages the lifecycle, preprocessing, and execution of the ML models."""
    def __init__(self):
        self.preprocessor = None
        self.model = None

    def load_artifacts(self):
        """Loads the pickle files into memory cleanly."""
        if not os.path.exists(settings.PREPROCESSOR_PATH) or not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifacts not found. Check pathing.\n"
                f"Preprocessor: {settings.PREPROCESSOR_PATH}\nModel: {settings.MODEL_PATH}"
            )
        self.preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
        self.model = joblib.load(settings.MODEL_PATH)

    def predict_segment(self, payload: CustomerInput) -> dict:
        """Processes raw user data and returns a structured cluster output."""
        input_df = pd.DataFrame(
            [[payload.accountAgeDays, payload.paymentMethodAgeDays]],
            columns=["accountAgeDays", "paymentMethodAgeDays"]
        )
        
        # Preprocess and scale data
        scaled_data = self.preprocessor.transform(input_df)
        
        # Predict the numerical cluster label
        probabilities = self.model['model'].predict_proba(scaled_data)[:, 1]
        predictions = (probabilities >= self.model['threshold']).astype(int).item()

        
        # Business logic mapping (matching your original app.py mapping)
        mapping = {
            0: "Non Fraud Transactions",
            1: "Fraud Transactions"
        }
        
        # Defensive check for unexpected cluster outputs
        transactions = mapping.get(predictions, "Unknown Preediction")
        
        return {"prediction_class": predictions, "prediction": transactions}

# Instantiate our global predictor instance
predictor = SegmentationPredictor()


# ------------------------------------------------------------------
# 2. THE FASTAPI APPLICATION LIFECYCLE
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup/shutdown operations. In production, we load heavy models
    ONCE here so the API can handle lightning-fast inference requests later.
    """
    try:
        predictor.load_artifacts()
        print("🎉 Machine Learning models loaded successfully into API runtime!")
    except Exception as e:
        print(f"❌ Critical Error loading ML models: {str(e)}")
        # In a real environment, force system crash if dependencies are missing
        raise e
    yield
    # Any cleanup code (like closing DB pools) would go here after yield
    print("Shutting down API...")


# Initialize the core FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan
)


# ------------------------------------------------------------------
# 3. ENDPOINT ROUTES
# ------------------------------------------------------------------
@app.get("/", tags=["Health Check"])
async def root():
    """A standard health-check endpoint to verify the server is alive."""
    return {"status": "healthy", "message": f"Welcome to {settings.APP_NAME}"}


@app.post(
    "/predict", 
    response_model=PredictionOutput, 
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
async def predict(payload: CustomerInput):
    """
    Accepts customer profile data, scales it, passes it through the 
    K-Means engine, and returns the computed segment classification.
    """
    try:
        result = predictor.predict_segment(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )