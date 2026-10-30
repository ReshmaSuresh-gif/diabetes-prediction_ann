import pandas as pd
from data_preprocessing import FEATURES


def predict_risk(model, values: dict) -> tuple[float, str]:
    """Return (probability of diabetes, risk band) for one patient."""
    row = pd.DataFrame([values])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    band = "Low" if p < 0.30 else "Moderate" if p < 0.60 else "High"
    return p, band
