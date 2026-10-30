import json
import joblib
from data_preprocessing import ROOT


def load_model():
    return joblib.load(ROOT / "models" / "ann_pipeline.joblib")


def load_meta():
    with open(ROOT / "models" / "meta.json") as f:
        return json.load(f)
