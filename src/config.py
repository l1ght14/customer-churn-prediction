"""Paths and shared constants. Single place to change anything global."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
SCORED_CSV = ROOT / "data" / "processed" / "churn_scored.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_JSON = ROOT / "reports" / "metrics.json"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# The 11 blank-TotalCharges rows are all tenure == 0 customers who never paid an
# invoice. Dropping them is correct; imputing would invent revenue that never happened.
BLANK_CHARGES_DROP_COUNT = 11

# Categorical predictors (everything left after the id, target and the numerics).
RAW_TEXT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
