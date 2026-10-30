"""Load, validate and clean the Pima Indians Diabetes dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "diabetes.csv"
CLEAN = ROOT / "data" / "processed" / "diabetes_cleaned.csv"

TARGET = "Outcome"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
# In this dataset a value of 0 is physiologically impossible for these columns,
# so it is a placeholder for "not recorded".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


if __name__ == "__main__":
    raw = load_raw()
    print("Raw shape:", raw.shape, "| duplicates:", raw.duplicated().sum())
    print("Zero counts:\n", (raw[ZERO_AS_MISSING] == 0).sum())
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN, index=False)
    print("Saved", CLEAN, cleaned.shape)
