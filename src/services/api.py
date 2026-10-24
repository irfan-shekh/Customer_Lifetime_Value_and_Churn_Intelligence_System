"""
FastAPI REST Inference Service.
Provides high-performance endpoints for health checks, real-time single customer scoring,
batch prediction, and CLV valuation.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.services.prediction_service import PredictionService
from src.database.connection import check_connection_health

app = FastAPI(
    title="Customer Lifetime Value & Churn Intelligence API",
    version="1.0.0",
    description="Production-grade REST API for Churn Risk Scoring, CLV Forecasts, and Prescriptive Recommendations.",
)

_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


# Pydantic Schemas
class CustomerFeatures(BaseModel):
    customer_id: Optional[str] = "CUST-PREDICT-01"
    gender: Optional[str] = "Female"
    senior_citizen: Optional[int] = Field(0, ge=0, le=1)
    partner: Optional[str] = "No"
    dependents: Optional[str] = "No"
    tenure: int = Field(..., ge=0, le=100, description="Customer tenure in months")
    phone_service: Optional[str] = "Yes"
    multiple_lines: Optional[str] = "No"
    internet_service: Optional[str] = "Fiber optic"
    online_security: Optional[str] = "No"
    online_backup: Optional[str] = "No"
    device_protection: Optional[str] = "No"
    tech_support: Optional[str] = "No"
    streaming_tv: Optional[str] = "Yes"
    streaming_movies: Optional[str] = "Yes"
    contract: str = Field(..., description="Month-to-month, One year, or Two year")
    paperless_billing: Optional[str] = "Yes"
    payment_method: Optional[str] = "Electronic check"
    monthly_charges: float = Field(..., ge=0.0, description="Monthly recurring charge in USD")
    total_charges: Optional[float] = Field(None, ge=0.0, description="Historical spend in USD")


class BatchCustomerRequest(BaseModel):
    customers: List[CustomerFeatures]


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint verifying model and database status."""
    db_health = check_connection_health()
    return {
        "api_status": "ONLINE",
        "database": {
            "status": db_health["status"],
            "storage_mode": db_health["storage_mode"],
            "message": db_health["message"],
        },
        "model_engine": "LightGBM + Lifelines + TreeExplainer",
    }


@app.post("/predict/customer", status_code=status.HTTP_200_OK)
def predict_single(customer: CustomerFeatures):
    """Generates real-time churn probability, risk tier, SHAP drivers, and CLV."""
    try:
        service = get_prediction_service()
        payload = customer.model_dump()
        if payload.get("total_charges") is None:
            payload["total_charges"] = payload["tenure"] * payload["monthly_charges"]
        return service.predict_single_customer(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", status_code=status.HTTP_200_OK)
def predict_batch(batch_req: BatchCustomerRequest):
    """Scores multiple customer records efficiently."""
    try:
        service = get_prediction_service()
        raw_list = []
        for c in batch_req.customers:
            d = c.model_dump()
            if d.get("total_charges") is None:
                d["total_charges"] = d["tenure"] * d["monthly_charges"]
            raw_list.append(d)
        return service.predict_batch(raw_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/clv/estimate", status_code=status.HTTP_200_OK)
def estimate_clv(
    monthly_charges: float,
    tenure: int,
    churn_probability: float = 0.25,
    horizon_months: int = 24,
    margin: float = 0.65,
    discount_rate: float = 0.10,
):
    """Calculates actuarial discounted CLV and expected future revenue."""
    try:
        service = get_prediction_service()
        res = service.clv_estimator.calculate_customer_clv(
            monthly_charges=monthly_charges,
            current_tenure=tenure,
            churn_probability=churn_probability,
            horizon_months=horizon_months,
            margin=margin,
            discount_rate=discount_rate,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLV calculation error: {str(e)}")
