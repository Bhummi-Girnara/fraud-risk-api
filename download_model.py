
import gdown

url = "https://drive.google.com/uc?id=16W-h_9LELbGq1sKNWQHJQ01I_vCRwvau"

gdown.download(
    url,
    "fraud_risk_model_v2_compressed.pkl",
    quiet=False
)
