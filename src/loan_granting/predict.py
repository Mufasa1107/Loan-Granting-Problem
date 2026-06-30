from typing import Any
import pandas as pd


def predict_loan_status(pipeline: Any, profile: dict[str, Any]) -> tuple[str, float]:
    features = pd.DataFrame([profile])
    probability = pipeline.predict_proba(features)[0][1]
    prediction = pipeline.predict(features)[0]
    decision = "Approved" if prediction == 1 else "Rejected"
    return decision, float(probability)
