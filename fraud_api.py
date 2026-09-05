
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AI Fraud Risk Prediction API")

# Load the retrained model
model = joblib.load("/content/fraud_risk_model_v2.pkl")
features = joblib.load("/content/fraud_model_features_v2.pkl")


class FraudInput(BaseModel):

    # Original 14 features
    issue_type: str
    transaction_channel: str
    transaction_amount: float
    failed_attempts: int
    device_mismatch: int
    location_mismatch: int
    new_beneficiary: int
    suspicious_url_flag: int
    qr_scam_flag: int
    otp_shared_flag: int
    unauthorized_flag: int
    account_compromise_flag: int
    pattern_flag: int
    repeated_complaints: int

    # Additional 9 features
    amount: float
    hour_of_day: int
    day_of_week: int
    is_round_amount: int
    amount_to_account_ratio: float
    previous_failed_attempts: int
    transaction_frequency_today: int
    is_new_recipient: int
    cross_state_transaction: int


@app.get("/")
def home():
    return {"message": "AI Fraud Risk Prediction API is running"}


@app.post("/predict")
def predict_fraud(data: FraudInput):

    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data[features]

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": "FRAUD" if prediction == 1 else "NOT_FRAUD",
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level
    }
