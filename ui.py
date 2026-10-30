import streamlit as st


def patient_form(meta):
    med, rng = meta["feature_medians"], meta["feature_ranges"]
    c1, c2 = st.columns(2)
    v = {}
    with c1:
        v["Pregnancies"] = st.number_input("Pregnancies", 0, 20, int(med["Pregnancies"]))
        v["Glucose"] = st.number_input("Plasma glucose (mg/dL)", 40, 250, int(med["Glucose"]))
        v["BloodPressure"] = st.number_input("Diastolic blood pressure (mm Hg)", 20, 140, int(med["BloodPressure"]))
        v["SkinThickness"] = st.number_input("Triceps skin-fold thickness (mm)", 5, 100, int(med["SkinThickness"]))
    with c2:
        v["Insulin"] = st.number_input("2-hour serum insulin (mu U/ml)", 10, 900, int(med["Insulin"]))
        v["BMI"] = st.number_input("BMI", 10.0, 70.0, float(med["BMI"]), step=0.1)
        v["DiabetesPedigreeFunction"] = st.number_input("Diabetes pedigree function", 0.05, 2.5,
                                                         float(med["DiabetesPedigreeFunction"]), step=0.01)
        v["Age"] = st.number_input("Age (years)", 18, 100, int(med["Age"]))
    return v
