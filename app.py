import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import streamlit as st
from model_loader import load_model, load_meta
from prediction import predict_risk
from ui import patient_form

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.caption("Artificial Neural Network (MLP) trained on the Pima Indians Diabetes dataset. "
           "Academic prototype - not a medical diagnosis.")


@st.cache_resource
def _load():
    return load_model(), load_meta()


model, meta = _load()
values = patient_form(meta)
if st.button("Predict risk", type="primary"):
    p, band = predict_risk(model, values)
    st.metric("Estimated probability of diabetes", f"{p:.1%}")
    {"Low": st.success, "Moderate": st.warning, "High": st.error}[band](f"Risk band: {band}")
    st.progress(min(max(p, 0.0), 1.0))
    st.info("This estimate reflects patterns in a small research dataset of adult women of Pima "
            "heritage. Consult a clinician for any health decision.")
